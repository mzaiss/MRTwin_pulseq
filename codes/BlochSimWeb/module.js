
// Jquery support of ES6 differs from other imported modules, and fails:
//   var $ = require( "https://unpkg.com/jquery/dist/jquery.js" );﻿  or
//   import {$,jQuery} from "https://unpkg.com/jquery/dist/jquery.js";
//   window.$ = $; // Expose to other imports.
//   window.jQuery = jQuery; // Expose to other imports.
// It requires extra require-module and becomes messy.

import * as dat from "https://unpkg.com/dat.gui/build/dat.gui.module.js";

// Apparantly selective modular import makes no difference compared to
import * as THREE from "https://unpkg.com/three@0.122.0/build/three.module.js";
// This list of functions seem to play no role, and size is independent.
// It is commented out since THREE otherwise have to be removed, except
// in webpack-version.
// import { Vector2, Vector3, Color, MeshBasicMaterial, MeshLambertMaterial,
// 	     Matrix4, Quaternion, CylinderBufferGeometry, Mesh, CircleGeometry,
// 	     PlaneGeometry, PlaneBufferGeometry, CircleBufferGeometry,
// 	     PerspectiveCamera, AmbientLight, DirectionalLight,
// 	     DirectionalLightHelper, FontLoader, TextBufferGeometry,
// 	     WebGLRenderer, Scene, AxisHelper} from "https://unpkg.com/three/build/three.module.js";

import { OrbitControls } from "https://unpkg.com/three@0.122.0/examples/jsm/controls/OrbitControls.js"; //imports from three.module.js, so only get Three from there.
// import { WEBGL } from "https://unpkg.com/three/examples/jsm/capabilities/WebGL.js";
import { WEBGL } from "https://unpkg.com/three@0.122.0/examples/jsm/WebGL.js";

"use strict"; // strict mode to avoid implicit globals added to global object "window".
// Variables declared here are added to "window". Prefer variables local to app.

var camera, controls, renderer;
var offsetHeight = 0;
var zoomFactor = 1;
var resizeTimeout;

function adjustToScreen() {

    let scrHeight = window.innerHeight;
    let scrWidth = window.innerWidth;

    let shiftUp = false;

    var appFlag = document.URL.indexOf('http://') === -1 &&
        document.URL.indexOf('https://') === -1; // Is code running as app?

    //	let AndroidDevice = navigator.userAgent.includes('Android'); //not IE compat
    let AndroidDevice = (navigator.userAgent.indexOf('Android') != -1);
    let iOSdevice = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
    if (AndroidDevice || iOSdevice) {
        if ((scrWidth < 800) && (scrWidth < scrHeight)) {
            $("#dialogUseLandscape").dialog({
                modal: false,
                buttons: {
                    Ok: function () {
                        $(this).dialog("close");
                    }
                }
            }).css("font-size", "35px");
            shiftUp = true;
        }
        else {
            $("#dialogUseLandscape:visible").dialog("close");
            shiftUp = false;
        }
        if (!appFlag) {
            $("#dialogConsiderApp").dialog({
                modal: false,
                minWidth: 500,
                buttons: {
                    Ok: function () {
                        $(this).dialog("close");
                    }
                }
            }).css("font-size", "35px");
        }
    }

    let totalWidth = 5; //5 is left margin
    $('.EventButtons').each(function () { //doesnt work if $() is cached
        offsetHeight || (offsetHeight = this.offsetHeight); //once!
        $(this).css({
            left: totalWidth
        });
        totalWidth += this.offsetWidth + 10;
    });
    totalWidth -= 5; // Reduce right margin

    if ((scrWidth < 800) || (totalWidth > scrWidth))
        zoomFactor = scrWidth / totalWidth;
    else
        zoomFactor = 1;

    $("#EventMenu").css('zoom', zoomFactor);

    $('.EventButtons').css('z-index', 20).each(function () {
        $(this).css({
            top: scrHeight / zoomFactor - (1 + shiftUp) * offsetHeight - 5
        });
    });

    let iconWidth = $(".icons").width();
    $(".icons").css({
        bottom: null,
        right: null,
        top: (scrHeight * 0.65) + "px",
        left: (scrWidth * 0.97 - iconWidth - 2) + "px" //requires width to be set in HTML.
    });

    if (scrHeight < 600) {
        $("#Saving").hide();
    } else {
        $("#Saving").show();
    }

    //let socMediaWidth = $( "#socialMedia" ).width()); //too small.
    if (scrWidth > (800 + 180)) {
        $("#socialMedia").show()
        if (scrWidth > (800 + 300))
            $("#specificSocialMedia").show()
        else
            $("#specificSocialMedia").hide();
    }
    else
        $("#socialMedia").hide();

    camera.aspect = scrWidth / scrHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(scrWidth, scrHeight);

    if ($("#newBlochSimulator").dialog("instance") &&
        $("#newBlochSimulator").dialog("isOpen")) { //reopen to center.
        $("#newBlochSimulator").dialog("close").dialog("open");
    }

} // adjustToScreen

function onResize() {
    // Throttle resizing. Ignore resize events as long as an adjustToScreen is queued.
    if (!resizeTimeout) {
        resizeTimeout = setTimeout(function () {
            resizeTimeout = null;
            adjustToScreen();
            window.setTimeout(0); //flush cache;
            adjustToScreen(); // fire twice since fast width changes may cause small errors.
            // The adjustToScreen() will execute at a rate of max 5fps
        }, 200);
    }
} // onResize

function dialog(id) {
    return function () {
        $("#" + id).dialog({
            modal: false//,
            // buttons: {
            // 	Ok: function() {
            // 	    $( this ).dialog( "close" );
            // 	}}

        });
    }
}

function colorExists(color) { // Checks if a color name is valid.
    if (color == 'white') {
	return true;
    }
    $('#colortest').css('backgroundColor', 'white'); // set a div color to white
    var whiteStr = $('#colortest').css('backgroundColor'); // return representation of white
    $('#colortest').css('backgroundColor', color); // change color
    return (!($('#colortest').css('backgroundColor') == whiteStr)); // has color changed from white?
}

function launchApp() { // started onload


    // Globals:  TODO: group them

    var debug = false;
    //
    // Set both reloadSceneResetsParms and hideWhenSelected false to provide two
    // levels of scene reload (minimal & full). Or latter true to save screen space:
    var reloadSceneResetsParms = false; // Scene main button does full reset.
    var hideWhenSelected = false; // Chosen scene label is removed from scene submenu.
    // Set Equilibrium menu item visible in HTML part if hideWhenSelected chosen false.
    //
    var scene;
    var eventButtons = $('.EventButtons'); //cache for speed. 
    var gui, state, blochMenu;
    var dt, lastTime, elapsed, then;
    const fpsInterval = 1000 / 70; //set at 30 to keep more constant rate or free CPU.
    // Will never be more than 60 (writing 70 makes it 60, if possible).
    const Tmax = 10; // maxmum finite relaxation time
    const B0max = 6;
    var savedState = {}; // used to save common parameters.
    var savedState2 = []; // used to save isochromate's parameters.
    var savedFlag = false; // is restoring of state an option?
    var paused = false;
    var guiFolderStrs = []; //stack of folder-names
    var guiFolders = [];    //stack of folders
    var addSampleFolder = false; //menu item for samples may confuse, so hidden.
    var nFolder = -1;
    var updateMenuList = [];
    var guiViewsFolder; // index of Fields menu item.
    var guiFieldsFolder; // index of Fields menu item.
    var guiGradientsFolder; // index of Gradient menu item.
    var guiFolderFlags = [true, true, true, true, true, true, true, true, true]; //folder closed?
    const guiUpdateInterval = 0.1; // seconds
    var guiTimeSinceUpdate = 0;

    const nullvec = new THREE.Vector3(0., 0., 0.);
    const unitXvec = new THREE.Vector3(1., 0., 0.);
    const unitYvec = new THREE.Vector3(0., 1., 0.);
    const unitZvec = new THREE.Vector3(0., 0., 1.);


    var GMvec = new THREE.Vector3(0., 0., 0.);
    var G_ADC = new THREE.Vector3(0., 0., 0.);

    var curveBlue = [], curveBlueTimes = [];
    var curveGreen = [], curveGreenTimes = [];
    var MxCurve = [], MxTimes = [];
    var MxyCurve = [], MxyTimes = [];
    var MzCurve = [], MzTimes = [];

    var RFCurve = [], RFTimes = [];
    var GxCurve = [], GxTimes = [];
    var GyCurve = [], GyTimes = [];
    var GadcCurve = [], GadcTimes = [];

    var MxLabelIdent = $('#MxLabel');
    var MxyLabelIdent = $('#MxyLabel');
    var MzLabelIdent = $('#MzLabel');

    var FID_backGround = fidbox;

    var FIDcanvas = document.getElementById("FIDcanvas");
    var FIDcanvasAxis = document.getElementById("FIDcanvasAxis");
    var FIDctx = FIDcanvas.getContext("2d");
    var FIDctxAxis = FIDcanvasAxis.getContext("2d");
    var grWidth = FIDcanvas.width;
    FIDcanvas.height = grWidth; // Needed to get aspect ratio of voxels right.
    FIDcanvasAxis.height = grWidth; // Apparantly canvas has two heights (clientheight)
    var grHeight = FIDcanvas.height;

    const FIDduration = 4000; //ms


    var RFLabelIdent = $('#RFLabel');
    var GxLabelIdent = $('#GxLabel');
    var GyLabelIdent = $('#adcLabel');

    var isRunningSequence = false; // run sequence flag
    var eventCache = [];

    var GMcanvas = document.getElementById("GMcanvas");
    var GMcanvasAxis = document.getElementById("GMcanvasAxis");
    var GMctx = GMcanvas.getContext("2d");
    var GMctxAxis = GMcanvasAxis.getContext("2d");
    // var grWidth = GMcanvas.width;
    GMcanvas.height = grWidth; // Needed to get aspect ratio of voxels right.
    GMcanvasAxis.height = grWidth; // Apparantly canvas has two heights (clientheight)
    // var grHeight = GMcanvas.height;

    const GMduration = 4000; //ms

    var trigSampleChange = false;
    var lastB1freq = 0;
    var delayB1vecUpdate = 0; // used to delay B1-updating a number of frames.
    var dtTotal = 0, dtCount = 0;
    //	var dtMemory = Array(10).fill(0), dtMemIndi = 0; // not IE compat
    var dtMemory = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtMemIndi = 0;
    var spoilR2 = 0;

    var scenes = []; // Array of functions defining scenes.

    var floor, floorRect, floorCirc;
    var B1cyl, B1shadow;

    var frameFixed = false; //Frame is fixed when spatial axis is active.

    var framePhase = 0; //phase of rotating frame
    var framePhase0 = 0; //phase of rotating frame at start of RF pulse.

    var statsContainer, stats;
    const torqueScale = 0.5;
    const gradScale = 11;
    const B1scale = 0.4;
    const spoilDuration = 1000;
    var spoilTimer1, spoilTimer2, spoilTimer3;
    var restartRepIfSampleChange = false;
    var restartRepIfSampleChangeTimer;
    var exciteTimers = [];

    const white = new THREE.Color('white');
    // Do not just use hex instead of color names. Colors: https://www.w3schools.com/colors/colors_groups.asp
    var greenList = ['lawngreen', 'chartreuse', 'green']; // preferred colors may not exist
    var blueList = ['dodgerblue','mediumblue','blue']; // preferred colors may not exist
    var color;
    do { color = greenList.shift() } while (!colorExists(color));
    const greenStr  = color;
    do { color = blueList.shift() } while (!colorExists(color));
    const blueStr  = color;
    
    const red = new THREE.Color('red');
    const green = new THREE.Color(greenStr); //only this can be chosen freely.
    const blue = new THREE.Color(blueStr);

    const nZeroSinc = 4; // 4 for 3-lobe sinc. Matching 0.22571 appears below.
    // A sinc with same B1 and duration as rect has durCorrSinc area ratio.
    // This is compensated by prolonging sinc pulse below. Wolfram knows Si(x).
    const durCorrSinc = 0.22571; // Si(2 pi)/(2 pi). Generally Si(nZ/2 pi) / (nZ/2 pi).

    const radius = 0.03; //cylinder radius
    const myShadow = true; // Shadows drawn manually.

    var shadowMaterial = new THREE.MeshBasicMaterial({ color: 0x808070 });
    var shadowMaterials = [];
    const downViewThresh = Math.PI / 4; // shadow decreases below 45 degree polar.

    // Semi-transparant test:
    //      var shadowMaterial = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true,  opacity: 0.5, blending: THREE.NormalBlending});
    var torqueMaterial = new THREE.MeshLambertMaterial({ color: 0xc80076 });
    var B1effMaterial = new THREE.MeshLambertMaterial({ color: "blue" });
    var floorMaterial = new THREE.MeshLambertMaterial({ color: 0xb0b090 }); //s:0x808070
    //	var floorMaterialFixed = new THREE.MeshLambertMaterial({ color: 0x90b0a0 });
    var floorMaterialFixed = new THREE.MeshLambertMaterial({ color: 0x90b0d0 });
    var floorMaterialBlack = new THREE.MeshLambertMaterial({ color: 0x303030 });
    const nShadowColors = 16;

    var doStats = false; // framerate statistics
    var addAxisHelper = false;
    var threeShadow = false; // Let Three.js handle shadows.

    var MaxGMvec = 1;
    // x-rotation 90 for testing:
    var propagator90x = new THREE.Matrix4().set(
        1, 0, 0, 0,
        0, Math.cos(Math.PI / 2), Math.sin(Math.PI / 2), 0,
        0, -Math.sin(Math.PI / 2), Math.cos(Math.PI / 2), 0,
        0, 0, 0, 1);

    var propagator90y = new THREE.Matrix4().set(
        Math.cos(Math.PI / 2), 0, Math.sin(Math.PI / 2), 0,
        0, 1, 0, 0,
        -Math.sin(Math.PI / 2), 0, Math.cos(Math.PI / 2), 0,
        0, 0, 0, 1);

    function cylinderMesh(fromVec, toVec, material, nElem, radius) {
        var vec = toVec.clone().sub(fromVec);
        var h = vec.length();
        vec.divideScalar(h || 1);
        var quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), vec);
        var geometry = new THREE.CylinderBufferGeometry(radius, radius, h, nElem);
        // BufferGeometries (eg. CylinderBufferGeometry) are faster
        // than Geometries and require less memory. See
        // https://threejsfundamentals.org/threejs/lessons/threejs-custom-buffergeometry.html
        geometry.translate(0, h / 2, 0);
        var cylinder = new THREE.Mesh(geometry, material);
        cylinder.applyQuaternion(quaternion);
        cylinder.position.set(fromVec.x, fromVec.y, fromVec.z);
        return cylinder;
    } //cylinderMesh

    function shadowMesh(Mvec) {
        var MvecTrans = new THREE.Vector2(Mvec.x, Mvec.y);
        var MvecTransLength = MvecTrans.length();

        var direction = Mvec.clone().projectOnPlane(unitZvec);

        var orientation = new THREE.Matrix4();
        orientation.lookAt(nullvec, Mvec.projectOnPlane(unitZvec), new THREE.Object3D().up);
        orientation.multiply(new THREE.Matrix4().set(1, 0, 0, 0,
            0, 0, 1, 0,
            0, -1, 0, 0,
            0, 0, 0, 1));

        var shadowBarGeo = new THREE.PlaneGeometry(2 * radius, MvecTransLength);
        shadowBarGeo.translate(0, MvecTransLength / 2, -1.1);
        var shadowBarMesh = new THREE.Mesh(shadowBarGeo);

        var shadowEndGeo = new THREE.CircleGeometry(radius, 4, 0, Math.PI); //only dot at far end.
        shadowEndGeo.translate(0, MvecTransLength, -1.1);
        var shadowEndMesh = new THREE.Mesh(shadowEndGeo);
        shadowEndMesh.rotation.z = MvecTrans.angle() + Math.PI / 2;

        var shadowGeo = new THREE.Geometry();  //merged BufferGeometry shadows fail to render with no
        // messages issued. Added Buffer here and above. https://stackoverflow.com/questions/36450612
        shadowGeo.merge(shadowBarMesh.geometry, shadowBarMesh.matrix);
        shadowGeo.merge(shadowEndMesh.geometry, shadowEndMesh.matrix);
        var mesh = new THREE.Mesh(shadowGeo, shadowMaterial);

        mesh.applyMatrix4(orientation);

        return mesh;
    } //shadowMesh

    function Isoc(M, color, pos, nElem, showCurve, dR1, dR2, M0, dRadius) {  //isocromate constructor
        this.M = M.clone();
        this.dB0 = 0; //spatially independent field offset
        this.detuning = 0; // total field offset from RF freq
        this.dMRF = new THREE.Vector3(0, 0, 1);
        this.color = color; // don't clone since color-pointer is used as identifier.
        this.pos = pos.clone();
        this.showCurve = showCurve;
        this.dR1 = dR1;
        this.dR2 = dR2;
        this.M0 = (M0 >= 0) ? M0 : 1;
        this.dRadius = dRadius ? dRadius : 0;


        nElem || (nElem = 8); // controls cylinder surface smoothness.TODO: reduce for small screen.
        var cylMaterial = new THREE.MeshLambertMaterial({ color: color });
        this.cylMesh = cylinderMesh(new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 1, 0),
            cylMaterial, nElem, radius + this.dRadius);
        scene.add(this.cylMesh);
        this.torque = cylinderMesh(new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 1, 0),
            torqueMaterial, nElem, radius + this.dRadius);
        scene.add(this.torque);
        this.B1eff = cylinderMesh(new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 1, 0),
            B1effMaterial, nElem, 1.01 * (radius + this.dRadius));
        scene.add(this.B1eff);

        if (myShadow) {
            // Shadows are initialized along y to make length right subsequently.
            this.shadow = shadowMesh(new THREE.Vector3(0, 1, 0));
            scene.add(this.shadow);
            this.tshadow = shadowMesh(new THREE.Vector3(0, 1, 0));
            scene.add(this.tshadow);
        }
    } //Isoc

    // Use prototype to avoid copies for each instance.
    // Likely more manipulation should be in prototype.
    // https://medium.com/better-programming/prototypes-in-javascript-5bba2990e04b

    Isoc.prototype.scale = function (scalar) {  //adds method "scale" to Isoc prototype.
        this.M = this.M.multiplyScalar(scalar);
        this.dMRF = this.dMRF.multiplyScalar(scalar);
        return this;
    }

    Isoc.prototype.remove = function () {
        scene.remove(this.cylMesh);
        scene.remove(this.torque);
        scene.remove(this.B1eff);
        if (myShadow) {
            scene.remove(this.shadow);
            scene.remove(this.tshadow);
        }
    }

    function scaleIsocArr(isocArr, factor) {
        isocArr.forEach(function (item, index) { item.scale(factor) });
        return isocArr;
    }

    function removeIsocArr() {
        state.IsocArr.forEach(function (item, index) { item.remove() });
    }


    function guiAddFolder(StrClosed, StrOpen, AdderFct, cFolder, createFromFolder) {
        if (createFromFolder <= 0) { // does nothing unless index to start creating from is zero (or less)
            let folderLabel = guiFolderFlags[cFolder] ? StrClosed : StrOpen;
            guiFolderStrs.push(folderLabel);
            let guiFolder = gui.addFolder(folderLabel);
            guiFolders.push(guiFolder);
            AdderFct(guiFolder);
            if (createFromFolder < 0) {
                guiFolderFlags[cFolder] = true;
            }
            if (guiFolderFlags[cFolder]) {
                guiFolder.close();
                // last folder's state of openess decides viewing order of stacked elems:
                if (cFolder == nFolder)  //reset depth to normal if last folder is closed.
                    eventButtons.css('z-index', 20);
            }
            else {
                guiFolder.open();
                if (cFolder == nFolder - 1) //lower buttons if last folder is open.
                    eventButtons.css('z-index', 0); // for small displays.
            }
            return guiFolder;
        }
    } //guiAddFolder

    function guiCloseFolder(guiFolder) {
        // possibly useful for Help dat-gui-item that does not need label updating.
        guiFolder.close();
        guiFolderFlags[guiFolders.indexOf(guiFolder)] = true;
    } //guiCloseFolder

    function scaleMultiple(magVecs, scalar) {
        for (let i = 0; i < magVecs.length; i++)
            magVecs[i].scale(scalar);
    }


    function RFconst(B1, B1freq) {
        let phase = B1freq * state.tSinceRF - state.phi1 + framePhase0;
        return [new THREE.Vector3(B1 * Math.cos(phase), -B1 * Math.sin(phase), 0.), B1];
    }

    function RFsincWrapper(duration) { //wrapper needed to avoid recalc of duration.
        return function RFsinc(B1, B1freq) {
            let phase = B1freq * state.tSinceRF - state.phi1 + framePhase0;
            let sincArg = nZeroSinc * Math.PI * (state.tSinceRF / duration - 1 / 2);
            let envelope = (Math.abs(sincArg) > 0.01) ?
                (B1 * Math.sin(sincArg) / sincArg) : B1;
            return [new THREE.Vector3(envelope * Math.cos(phase),
                -envelope * Math.sin(phase), 0.),
                envelope];
        }
    }


    function RFpulse(type, angle, phase, B1) {
        let gamma = state.Gamma;
        state.tSinceRF = 0; // Both area and time left is needed for pulse with
        state.areaLeftRF = angle; // sidelobes. Area is adjusted at temporal end.
        let duration = angle / (gamma * B1);
        state.B1 = B1;
        state.B1freq = gamma * state.B0;
        var dtAvg = dtMemory.reduce(function (a, b) { return a + b }, 0) / dtMemory.length;  //short function notation is not IE compatible.

        phase += state.B1freq * gamma * dtAvg / 2; //small phase correction
        framePhase0 = framePhase;
        state.phi1 = phase;
        switch (type) {
            case 'rect': state.RFfunc = RFconst; break;
            case 'sinc': duration = duration / durCorrSinc;
                state.RFfunc = RFsincWrapper(duration); break;
            default: alert('Unknown RF pulse type');
        }
        state.tLeftRF = duration;
        updateMenuList.push(guiFieldsFolder); //mark field folder for updating
    }  //RFpulse

    function spoil() {
        spoilR2 = 4.7;
        window.setTimeout(
            function () {
                spoilR2 = 0;
                if (state.Sample == "Thermal ensemble") return; //dont spoil thermal.
                for (var i = 0; i < state.IsocArr.length; i++) { //kill any remaining Mxy.
                    state.IsocArr[i].M.projectOnVector(unitZvec);
                }
            },
            spoilDuration); //ms
    } //spoil

    function gradPulse(phaseDiff, directionAngle) {
        const gradDur = 1; //s
        state.areaLeftGrad = phaseDiff * gradScale / state.Gamma;
        if (directionAngle) {
            state.Gx = Math.cos(directionAngle) * state.areaLeftGrad / gradDur;
            state.Gy = Math.sin(directionAngle) * state.areaLeftGrad / gradDur;
        } else { //default is Gx
            state.Gx = state.areaLeftGrad / gradDur;
            directionAngle = 0;
        }
        state.PulseGradDirection = directionAngle;
        updateMenuList.push(guiGradientsFolder);
    } // gradPulse

    function gradRefocus() {
        let isocArr = state.IsocArr;
        let meanPhaseDiff = 0;
        let MxyLeft, MxyRight;
        let dx, weight;
        let totalWeight = 0;
        let phaseRight; // Right isocs phase
        let phaseDiff; //phase difference
        MxyLeft = isocArr[0].M.clone().projectOnPlane(unitZvec);
        for (let i = 1; i <= isocArr.length - 2; i++) {
            dx = isocArr[i].pos.x - isocArr[i - 1].pos.x;
            if (dx > 0) { // ignore lineshifts in plane
                MxyRight = isocArr[i].M.clone().projectOnPlane(unitZvec);
                weight = Math.min(MxyLeft.length(), MxyRight.length());
                totalWeight += weight;
                phaseRight = Math.atan2(MxyRight.y, MxyRight.x);
                MxyLeft.applyAxisAngle(unitZvec, -phaseRight); //rotate left isoc by right's angle.
                phaseDiff = Math.atan2(MxyLeft.y, MxyLeft.x);
                meanPhaseDiff += weight * phaseDiff / dx;
                MxyLeft = MxyRight; // right is the new left
            }
        }
        if ((Math.abs(meanPhaseDiff) > 0.001) && (totalWeight > 0.01)) {
            meanPhaseDiff = meanPhaseDiff / totalWeight;
            gradPulse(-meanPhaseDiff);
        }
    } // gradRefocus


    function thermalDrawFromLinearDist(B0) { //draws sample from -1 to 1
        const pol = B0 / B0max; // 0 to 1. Zero gives uniform distribution.
        var sample;

        let random = Math.random();
        if (random > pol) //sample from uniform dist, if random is gt pol-treshold.
            sample = 2 * Math.random() - 1
        else
            sample = 2 * Math.sqrt(Math.random()) - 1; //linear dist
        return sample;
    }

    function magInit() {
        const c10 = Math.cos(10 * Math.PI / 180);
        const s10 = Math.sin(10 * Math.PI / 180);
        const c30 = Math.cos(30 * Math.PI / 180);
        const s30 = Math.sin(30 * Math.PI / 180);
        const eps = 0.05;
        const xz30 = new THREE.Vector3(s30, 0, c30);
        const x = new THREE.Vector3(1, 0, 0);
        const y = new THREE.Vector3(0, 1, 0);
        const z = new THREE.Vector3(0, 0, 1);
        const nx = new THREE.Vector3(-1, 0, 0);
        const ny = new THREE.Vector3(0, -1, 0);
        const nz = new THREE.Vector3(0, 0, -1);
        const xyz = new THREE.Vector3(1 + eps, 1 - eps, 1);
        const xynz = new THREE.Vector3(1, 1, -1);
        const xnyz = new THREE.Vector3(1 + eps, -1 + eps, 1);
        const nxyz = new THREE.Vector3(-1 - eps, 1 - eps, 1);
        const nxnyz = new THREE.Vector3(-1 - eps, -1 + eps, 1);
        const xnynz = new THREE.Vector3(1, -1, -1);
        const nxynz = new THREE.Vector3(-1, 1, -1);
        const nxnynz = new THREE.Vector3(-1, -1, -1);

        function IsocXz30() {
            return new Isoc(xz30, // M
                white, // color
                nullvec);
        }  // pos
        function IsocX() { return new Isoc(x, white, nullvec); }
        function IsocY() { return new Isoc(y, white, nullvec); }
        function IsocZ() { return new Isoc(z, white, nullvec); }
        function IsocZensembleRed() { return new Isoc(z, red, nullvec); }
        let nElem = 8; //controls cylinder surface smoothness.
        function IsocZgreen() {
            let M0 = 0.91;
            return new Isoc(
                new THREE.Vector3(0, 0, M0),
                green, nullvec, nElem, // Note: added relax dR1, dR2 must be pos.
                true, 0, 0, M0, 0);
        } //showCurve, dR1, dR2, M0, dRadius
        function IsocZblue() {
            let M0 = 1.0;
            return new Isoc(
                new THREE.Vector3(0, 0, M0),
                blue, nullvec, nElem,
                true, 0.2, 0.2, M0, 0.001);
        } //showCurve, dR1, dR2, M0, dRadius

        function IsocZwhite() {
            let M0 = 0.91;
            return new Isoc(
                new THREE.Vector3(0, 0, M0),
                white, nullvec, nElem,
                true, 0, 0.2, M0, 0.0008);
        } //showCurve, dR1, dR2, M0, dRadius
        function IsocNX() { return new Isoc(nx, white, nullvec); }
        function IsocNY() { return new Isoc(ny, white, nullvec); }
        function IsocNZ() { return new Isoc(nz, white, nullvec); }
        function IsocXYZ() { return new Isoc(xyz, white, nullvec); }
        function IsocXYNZ() { return new Isoc(xynz, white, nullvec); }
        function IsocXNYZ() { return new Isoc(xnyz, white, nullvec); }
        function IsocNXYZ() { return new Isoc(nxyz, white, nullvec); }
        function IsocNXNYZ() { return new Isoc(nxnyz, white, nullvec); }
        function IsocXNYNZ() { return new Isoc(xnynz, white, nullvec); }
        function IsocNXYNZ() { return new Isoc(nxynz, white, nullvec); }
        function IsocNXNYNZ() { return new Isoc(nxnynz, white, nullvec); }

        let basicState = {
            IsocArr: [], B1: 0.0, Gamma: 1,
            B0: 2.,
            //			     t:0, //removed to avoid resetting of FID. Seems ok.
            dt: 0.01, phi1: 0., T1: Infinity, T2: Infinity, B1freq: 5,
            Name: '', RFfunc: RFconst
        };

        scenes.Isoc1 = function () {
            return { IsocArr: [IsocZ()] }
        };

        scenes.Precession = function () {
            Object.assign(state, basicState);
            return { IsocArr: [IsocXz30()] }
        };

        scenes.Equilibrium = function () {
            Object.assign(state, basicState);
            return {
                IsocArr: [IsocZ()],
                T1: Infinity, T2: Infinity,
                RFfunc: RFconst,
                viewMx: true
            };
        }

        scenes.IsocInhomN = function (nIsoc) {
            let inhom = { IsocArr: [] };
            const spreadScale = 1 / 6;
            const nonlinScale = Math.PI / 1.5; //reduces recovery
            // const spreadScale = 1 / 1;
            // const nonlinScale = Math.PI / 9; // change t2* for inhomo
            for (let i = 0; i < nIsoc; i++) {
                inhom.IsocArr.push(IsocZ());
                inhom.IsocArr[i].dB0 =
                    Math.tan((i - (nIsoc - 1) / 2) / (nIsoc / nonlinScale)) * spreadScale;
            }
            return inhom;
        } //IsocInhomN

        scenes.Inhomogeneity = function () {
            Object.assign(state, basicState);
            Object.assign(state, scenes.IsocInhomN(9));
            return {};
        }


        scenes.ThermalEnsemble = function () {
            // Creates pseudo random state that appears more random than random.
            // For each cosTheta, 3+-k*B0 isocs evenly rotated over cirle are added.
            let B0 = B0max;
            const nBand = 100; //select even
            const perBand = 3;
            var Isocs = [];
            var cosTheta;
            let M = new THREE.Vector3;
            for (let i = 0; i < nBand; i++) {
                cosTheta = i - nBand / 2 + 0.5; //symmetric and avoid extremes
                cosTheta = cosTheta / (nBand / 2); // -1 < cosTheta < 1
                let phi = Math.random() * 2 * Math.PI;
                let perBandAdjusted = Math.round((1 + cosTheta * (B0 / B0max)) * perBand);
                for (let j = 0; j < perBandAdjusted; j++) {
                    M.z = cosTheta;
                    let Mxy = Math.sqrt(1 - M.z * M.z);
                    let arg = phi + (2 * Math.PI) * (j + Math.random() / 2) / perBandAdjusted;
                    M.x = Mxy * Math.cos(arg);
                    M.y = Mxy * Math.sin(arg);
                    Isocs.push(new Isoc(M, white, nullvec));
                }
            }
            return {
                IsocArr: Isocs, viewMz: true, // B0:B0, (left low for better viz)
                FrameStat: false, FrameB0: true, FrameB1: false
            };
        } //ThermalEnsemble

        scenes.ThermalEnsembleSimple = function () { // NOT used currently.
            // simplified view. Fails when relaxation is added.
            let axisVecs = [IsocZensembleRed().scale(1.03), IsocNZ().scale(0.97),
            IsocX(), IsocY(), IsocNX(), IsocNY()];
            axisVecs[0].color = red;
            let diagVecs = [IsocXYZ(), IsocNXYZ(), IsocXNYZ(), IsocXYNZ(),
            IsocNXNYZ(), IsocNXYNZ(), IsocXNYNZ(), IsocNXNYNZ()];
            diagVecs = scaleIsocArr(diagVecs, 1 / Math.sqrt(3));
            return { IsocArr: axisVecs.concat(diagVecs) };

        } //ThermalEnsembleSimple


        scenes.Ensemble = function () {
            Object.assign(state, basicState);
            return scenes.ThermalEnsemble();
        }

        scenes.Substances3 = function () {
            let isocs = { IsocArr: [IsocZblue(), IsocZgreen(), IsocZwhite()] }; //
            isocs.IsocArr[0].dB0 = 0; 
            isocs.IsocArr[1].dB0 = -0.04;
            isocs.IsocArr[2].dB0 = 0.04;
            return isocs;
        }

        scenes.MixedMatter = function () {
            Object.assign(state, basicState);
            state.T1 = 8;
            state.T2 = 5;
            Object.assign(state, scenes.Substances3());
            return { viewMx: false, viewMxy: true };
        }

        scenes.Line = function () {
            const nIsoc = 21; // choose odd number of isochromates.
            var line = [];
            for (let i = 0; i < nIsoc; i++) {
                line.push(IsocZ());
                line[i].pos.setX((i - (nIsoc - 1) / 2) * 0.4);
            }
            return { IsocArr: line, allScale: 0.35, Gx: 3 };
        }

        // scenes.LineDense = function () {
        // 	const nIsoc = 41; // choose odd number of isochromates.
        // 	var line = [];
        // 	for(let i=0; i<nIsoc; i++) {
        // 	    line.push(IsocZ());
        // 	    line[i].pos.setX((i-(nIsoc-1)/2)*0.2);
        // 	}
        // 	return {IsocArr: line, allScale: 0.35};
        // }

        scenes.LineDense = function (uniform) {
            const nIsoc = 41; // Not all are realized for structured object.
            var line = [];
            var shift = 1;
            for (let i = 0; i < nIsoc; i++) {
                if (uniform || ((Math.floor((i - shift) / 3) % 2) == 0)) {
                    let isoc = IsocZ();
                    isoc.pos.setX((i - (nIsoc - 1) / 2) * 0.2);
                    line.push(isoc);
                }
            }
            return { IsocArr: line, allScale: 0.35, Gx: -6 };
        }

        scenes.Plane = function () {
            const nIsoc = 21; // choose odd number of isochromates.
            var plane = [];
            for (let i = 0; i < nIsoc; i++) {
                for (let j = 0; j < nIsoc; j++) {
                    plane.push(IsocZ());
                    plane[i * nIsoc + j].pos.set((j - (nIsoc - 1) / 2) * 0.4, (i - (nIsoc - 1) / 2) * 0.4, 0);
                }
            }
            return { IsocArr: plane, allScale: 0.35 };
        }

        scenes.WeakGradient = function () {
            Object.assign(state, basicState);
            Object.assign(state, scenes.Line());
            return { B1freq: 3 };
        }

        scenes.StrongGradient = function (uniform) {
            Object.assign(state, basicState);
            Object.assign(state, scenes.LineDense(uniform));
            return { B1freq: 0, FrameB0: true, FrameB1: false, FrameStat: false };
        }


    } //magInit

    function initFIDctxAxis() {
        FIDctxAxis.clearRect(0, 0, grWidth, grHeight);
        FIDctxAxis.save();
        FIDctxAxis.strokeStyle = 'gray';
        FIDctxAxis.fillStyle = 'gray';
        FIDctxAxis.lineWidth = 1;
        let offset = 4; //half triangle size
        FIDctx.translate(offset, grHeight / 2);
        FIDctx.scale(0.95, 0.95);
        FIDctx.translate(offset, -grHeight / 2);
        FIDctxAxis.beginPath();
        FIDctxAxis.moveTo(offset, 0); //vertical axis:
        FIDctxAxis.lineTo(offset, grHeight);
        let nTick = 8; // tick marks:
        for (let cTick = 1; cTick < nTick; cTick++) {
            FIDctxAxis.moveTo(-offset, grHeight * cTick / nTick);
            FIDctxAxis.lineTo(offset, grHeight * cTick / nTick);
        }
        FIDctxAxis.stroke();
        FIDctxAxis.beginPath(); // triangle:
        FIDctxAxis.moveTo(offset, 0);
        FIDctxAxis.lineTo(0, 2 * offset);
        FIDctxAxis.lineTo(2 * offset, 2 * offset);
        FIDctxAxis.fill();
        FIDctxAxis.beginPath();
        FIDctxAxis.moveTo(offset, grHeight / 2); //horizontal axis:
        FIDctxAxis.lineTo(grWidth, grHeight / 2);
        FIDctxAxis.stroke();
        FIDctxAxis.beginPath(); // triangle:
        FIDctxAxis.moveTo(grWidth - 2 * offset, grHeight / 2 - offset);
        FIDctxAxis.lineTo(grWidth, grHeight / 2);
        FIDctxAxis.lineTo(grWidth - 6, grHeight / 2 + offset);
        FIDctxAxis.fill();
        FIDctxAxis.restore();
    } //initFIDctxAxis

    function initGMctxAxis() {
        GMctxAxis.clearRect(0, 0, grWidth, grHeight);
        GMctxAxis.save();
        GMctxAxis.strokeStyle = 'gray';
        GMctxAxis.fillStyle = 'gray';
        GMctxAxis.lineWidth = 1;
        let offset = 4; //half triangle size
        GMctx.translate(offset, grHeight / 2);
        GMctx.scale(0.95, 0.95);
        GMctx.translate(offset, -grHeight / 2);
        GMctxAxis.beginPath();
        GMctxAxis.moveTo(offset, 0); //vertical axis:
        GMctxAxis.lineTo(offset, grHeight);
        let nTick = 8; // tick marks:
        for (let cTick = 1; cTick < nTick; cTick++) {
            GMctxAxis.moveTo(-offset, grHeight * cTick / nTick);
            GMctxAxis.lineTo(offset, grHeight * cTick / nTick);
        }
        GMctxAxis.stroke();
        GMctxAxis.beginPath(); // triangle:
        GMctxAxis.moveTo(offset, 0);
        GMctxAxis.lineTo(0, 2 * offset);
        GMctxAxis.lineTo(2 * offset, 2 * offset);
        GMctxAxis.fill();
        GMctxAxis.beginPath();
        GMctxAxis.moveTo(offset, grHeight / 2); //horizontal axis:
        GMctxAxis.lineTo(grWidth, grHeight / 2);
        GMctxAxis.stroke();
        GMctxAxis.beginPath(); // triangle:
        GMctxAxis.moveTo(grWidth - 2 * offset, grHeight / 2 - offset);
        GMctxAxis.lineTo(grWidth, grHeight / 2);
        GMctxAxis.lineTo(grWidth - 6, grHeight / 2 + offset);
        GMctxAxis.fill();
        GMctxAxis.restore();
    } //initGMctxAxis update 07.2022


    function sampleChange() {

        if (paused) {
            paused = false;
            $("#Pause").button("option", "label", "||");
        }

        trigSampleChange = false;  //clear request for further updating.
        removeIsocArr();
        state.allScale = 1; //default
        state.curveScale = 1; //default
        switch (state.Sample) {
            case 'Precession': /* Scene-changes from here: */
                state = Object.assign(state, scenes.Precession());
                state.Sample = '1 isochromate';
                frameFixed = false; //TODO: Why is frameFixed not in scenes-definitions?
                $("#Presets").css('color', '#bbbbbb'); break;
            case 'Equilibrium':
                state = Object.assign(state, scenes.Equilibrium());
                state.Sample = '1 isochromate';
                frameFixed = false;
                $("#Presets").css('color', '#ffffff'); break;
            case 'Inhomogeneity':
                state = Object.assign(state, scenes.Inhomogeneity());
                state.Sample = '9 isochromates';
                frameFixed = false;
                $("#Presets").css('color', '#ffffff'); break;
            case 'Ensemble':
                state = Object.assign(state, scenes.Ensemble());
                state.Sample = 'Thermal ensemble';
                state.curveScale = 2;
                frameFixed = false;
                $("#Presets").css('color', '#ffffff'); break;
            case 'Weak gradient':
                state = Object.assign(state, scenes.WeakGradient());
                frameFixed = true;
                state.Sample = 'Line';
                $("#Presets").css('color', '#ffffff'); break;
            case 'Strong gradient':
                state = Object.assign(state, scenes.StrongGradient(true));
                frameFixed = true;
                state.Sample = 'Line, dense';
                $("#Presets").css('color', '#ffffff'); break;
            case 'Structure':
                state = Object.assign(state, scenes.StrongGradient(false));
                frameFixed = true;
                state.Sample = 'Line, structured';
                $("#Presets").css('color', '#ffffff'); break;
            case 'Mixed matter':
                state = Object.assign(state, scenes.MixedMatter());
                state.Sample = '3 substances';
                frameFixed = false;
                $("#Presets").css('color', '#ffffff'); break; /* Sample-changes from here: */
            case '1 isochromate':
                state = Object.assign(state, scenes.Isoc1());
                frameFixed = false; break;
            case '9 isochromates':
                state = Object.assign(state, scenes.IsocInhomN(9));
                frameFixed = false; break;
            case '3 substances':
                state = Object.assign(state, scenes.Substances3());
                frameFixed = false; break;
            case 'Thermal ensemble':
                state = Object.assign(state, scenes.ThermalEnsemble());
                frameFixed = false; break;
            case 'Line':
                state = Object.assign(state, scenes.Line());
                frameFixed = true; break;
            case 'Line, dense':
                state = Object.assign(state, scenes.LineDense(true));
                frameFixed = true; break;
            case 'Line, structured':
                state = Object.assign(state, scenes.LineDense(false));
                frameFixed = true; break;
            case 'Plane':
                state = Object.assign(state, scenes.Plane());
                frameFixed = true; break;
            default: alert("Sample changed to " + state.Sample);
        }


        if (restartRepIfSampleChange) { // Re-excite is relevant for multi-SE.
            clearRepTimers();
            restartRepIfSampleChangeTimer = window.setTimeout(
                function () {
                    let elem = document.getElementById('RepExc');
                    let label = elem.textContent || elem.innerText || "";
                    buttonAction(label);
                }
                , 4000);
        }

        if (frameFixed && (!state.FrameStat))
            floor.material = floorMaterialFixed;
        else
            floor.material = floorMaterial;
        shadowMaterialsInit(floor.material);

        // Update FID label visibility:
        state.viewMx ? MxLabelIdent.show() : MxLabelIdent.hide();
        state.viewMz ? MzLabelIdent.show() : MzLabelIdent.hide();
        state.viewMxy ? MxyLabelIdent.show() : MxyLabelIdent.hide();
        state.viewRF ? RFLabelIdent.show() : RFLabelIdent.hide();
        state.viewGx ? GxLabelIdent.show() : GxLabelIdent.hide();
        state.viewGy ? GyLabelIdent.show() : GyLabelIdent.hide();
        curveBlue.forEach(function (item, index) { curveBlue[index] = 0 });
        curveGreen.forEach(function (item, index) { curveGreen[index] = 0 });

    } //sampleChange


    function guiInit(removeFolderArg) {
        // Initializes new gui, or removes&recreates from folder removeFolderArg for updating.
        // There may be alternatives for updating dat-gui, but this was made before knowing, e.g. so
        //	gui.__folders['Relaxation: Off'].__controllers[1].setValue(Infinity));
        // For details and helper functions see:
        //   https://stackoverflow.com/questions/16166440/refresh-dat-gui-with-new-values
        // My solution may be somewhat slow, and seems to prevent dat-gui presets from working well.

        var createFromFolder;
        debug && console.log('guiInit called. Argument: ' + (removeFolderArg ? removeFolderArg : ''));
        if (!gui) { // if new gui

            blochMenu = {
                GetStarted: dialog("dialogGetStarted"),
                VideoIntros: dialog("dialogVideoIntros"),
                GetApps: dialog("dialogGetApps"),
                Tools: dialog("dialogTools"),
                About: dialog("dialogAbout"),
                Reset: function () { trigSampleChange = true; }
            }

            state = { //dummy example values. More are added for some samples.
                B0: 0,
                B1: 0,
                B1freq: 0, //angular frequency
                phi1: 0,  //RF phase in rotating frame.
                T1: 1, T2: 1,
                Gx: 0, Gy: 0,
                viewB1: false, viewTorqB1eff: true,
                viewMxy: true, viewMx: true, viewMz: false, viewB1x: false,
                viewRF: true, viewGx: true, viewGy: true,
                FrameStat: true, FrameB0: false, FrameB1: false,
                Sample: '1 isochromate',
                allScale: 1,
                RFfunc: RFconst,
                IsocArr: [],
                t: 0, tSinceRF: 0,
                RF: 1, GM: 1,
                Gamma: 1,
            };

            gui = new dat.GUI({ autoPlace: false });
            var customContainer = $('.moveGUI').append($(gui.domElement));
            // $(gui.domElement).resizable(); //Not as intended, but has potential.
            // Initial CSS is overwritten by "new". Either move CSS
            // down or change dynamically as exemplified:
            // $('head').append(`<style>.dg li { z-index: 10; }</style>`);
            // This works, but late CSS is better. Keep example here.

            createFromFolder = 0;

        } else { //if existing gui needing updating since folder opened/closed:
            createFromFolder = guiFolderStrs.length;
            let popped;
            do {
                popped = guiFolderStrs.pop();  //remove last created folders until relevant folder is reached.
                gui.removeFolder(guiFolders.pop());
                createFromFolder--;
            } while ((popped != removeFolderArg) && (createFromFolder >= 0));
        }

        var cFolder = 0;

        // All folders are attempted added, even if preexisting. However, guiAddFolder
        // doesn't do anything until index createFromFolder is reached (indicates
        // first folder to create, counting from cFolder).    
        //below: Insight at https://stackoverflow.com/questions/30372761/map-dat-gui-dropdown-menu-strings-to-values
        guiAddFolder('Bloch Simulator',
            'Bloch Simulator',
            function (guiFolder) {
                guiFolder.domElement.style.fontWeight = "bold";
                guiFolder.domElement.style.backgroundColor = "transparent";
                let tmp = guiFolder.add(blochMenu, "GetStarted").name("Get started");
                tmp.domElement.style.fontWeight = "normal"; //prop exist but no 
                // effect indep of open/close/displayUpdate/order... dat-gui bug?
                guiFolder.add(blochMenu, "VideoIntros").name("Video intros");
                guiFolder.add(blochMenu, "GetApps").name("Get or rate app");
                guiFolder.add(blochMenu, "Tools").name("Related tools");
                guiFolder.add(blochMenu, "About");
            },
            cFolder++, createFromFolder--);

        if (addSampleFolder) {
            guiAddFolder('Sample: ' + state.Sample,
                'Sample',
                function (guiFolder) {
                    let tmp = state.Sample; //state.Sample is inadvertently changed by folder creation. Weirdly, the temporary saving of state cannot be moved to guiAddFolder. General problem.
                    guiFolder.add(state, 'Sample', savedFlag ? [state.Sample] :
                        ['1 isochromate',
                            '7 isochromates',
                            '3 substances',
                            'Line',
                            'Line, dense',
                            'Periodic',
                            'Thermal ensemble',
                            'Plane']
                    ).//listen(). //only listen() when external changes needs attention
                        onChange(function () { trigSampleChange = true; });
                    guiFolder.add(blochMenu, 'Reset');
                    state.Sample = tmp;
                },
                cFolder++, createFromFolder--);
        }

        let relaxStr;
        if ((state.T1 == Infinity) && (state.T2 == Infinity))
            relaxStr = 'Off'
        else if (state.T1 == Infinity)
            relaxStr = 'T1 off, T2=' + state.T2
        else
            relaxStr = 'T1=' + state.T1 + ', T2=' + state.T2;

        guiAddFolder('Relaxation: ' + relaxStr,
            'Relaxation',
            function (guiFolder) {
                let tmp2 = state.T1, tmp3 = state.T2;
                // .listen() is needed to ensure T2<T1, but disables text
                // input without patching dat.gui: "Allows NumberSlider and
                // Box to be edited during .listen()" (github page):
                guiFolder.add(state, 'T1', 0, Tmax, 1).listen().onChange(
                    function () {
                        if (state.T1 == Tmax) {
                            this.updateDisplay(); //moves slider to max.
                            state.T1 = Infinity;
                        }
                        if (state.T2 > state.T1)
                            state.T2 = state.T1
                    });
                // .listen() is needed to ensure T1>T2. More details above.
                guiFolder.add(state, 'T2', 0, Tmax, 1).listen().onChange(
                    function () {
                        if ((state.T1 < state.T2) && (spoilR2 == 0))
                            state.T1 = state.T2;
                        if (state.T2 == Tmax) {
                            this.updateDisplay(); //move T2 slider to max.
                            this.__gui.__controllers[0].updateDisplay(); //T1 slider also
                            state.T2 = Infinity;
                            state.T1 = Infinity;
                        }
                    });
                state.T1 = tmp2; state.T2 = tmp3;
            },
            cFolder++, createFromFolder--);

        guiViewsFolder = cFolder; // folder index is needed to update FID labels.

        let viewStr = (state.viewB1 ? 'B1,' : '') +
            ((state.viewTorqB1eff && (!state.FrameB1)) ? 'Torque,' : '') +
            ((state.viewTorqB1eff && (state.FrameB1)) ? 'B1eff,' : '') +
            (state.viewMx ? 'Mx,' : '') +
            (state.viewMxy ? '|Mxy|,' : '') +
            (state.viewMz ? 'Mz,' : '') +
            (state.viewRF ? 'RF,' : '') +
            (state.viewGx ? 'Gx,' : '') +
            (state.viewGy ? 'Gy,' : '');
        viewStr = viewStr.slice(0, -1); //chop
        guiAddFolder('View: ' + viewStr,
            'View',
            function (guiFolder) {
                let tmp1 = state.viewB1,
                    tmp2 = state.viewTorqB1eff,
                    tmp3 = state.viewMx,
                    tmp4 = state.viewMxy,
                    tmp5 = state.viewMz,
                    tmp6 = state.viewRF,
                    tmp7 = state.viewGx,
                    tmp8 = state.viewGy;
                guiFolder.add(state, 'viewB1').name('B1');
                guiFolder.add(state, 'viewTorqB1eff').
                    name('Torque / B1eff   ');
                    guiFolder.add(state, 'viewMx').name('Mx').
                        onChange(function () {
                            if (state.viewMx)
                                MxLabelIdent.show()
                            else
                                MxLabelIdent.hide()
                        });
                    guiFolder.add(state, 'viewMxy').name('|Mxy|').
                        onChange(function () {
                            if (state.viewMxy)
                                MxyLabelIdent.show()
                            else
                                MxyLabelIdent.hide()
                        });
                    guiFolder.add(state, 'viewMz').name('Mz').
                        onChange(function () {
                            if (state.viewMz)
                                MzLabelIdent.show()
                            else
                                MzLabelIdent.hide()
                        });
                    guiFolder.add(state, 'viewRF').name('RF').
                        onChange(function () {
                            if (state.viewRF)
                                RFLabelIdent.show()
                            else
                                RFLabelIdent.hide()
                        });
                    guiFolder.add(state, 'viewGx').name('Gx/Gy').
                        onChange(function () {
                            if (state.viewGx)
                                GxLabelIdent.show()
                            else
                                GxLabelIdent.hide()
                        });
                    guiFolder.add(state, 'viewGy').name('adc').
                        onChange(function () {
                            if (state.viewGy)
                                GyLabelIdent.show()
                            else
                                GyLabelIdent.hide()
                        });
                state.viewB1 = tmp1;
                state.viewTorqB1eff = tmp2;
                state.viewMx = tmp3;
                state.viewMxy = tmp4;
                state.viewMz = tmp5;
                state.viewRF = tmp6;
                state.viewGx = tmp7;
                state.viewGy = tmp8;
            },
            cFolder++, createFromFolder--);

        guiFieldsFolder = cFolder; // folder index is needed for updating during RF pulses.
        guiAddFolder('Fields: B0=' + state.B0 +
            ', B1=' + Math.round(state.B1 * 10) / 10 +
            ', B1freq=' + state.B1freq, //subscript doesnt work. No RF of B1 subscripts exist.
            'Fields',
            function (guiFolder) {
                let tmp1 = state.B0, tmp2 = state.B1, tmp3 = state.B1freq;
                guiFolder.add(state, 'B0', 0, B0max, 1);
                guiFolder.add(state, 'B1', 0, 3, 0.3);
                guiFolder.add(state, 'B1freq', 0, B0max, 0.5).//listen().
                    onChange(
                        function () { //Frame and B1 phase must be continuous.
                            if (frameFixed) { return; } //B1freq is zero then.
                            let B1freq = state.B1freq;
                            // in RFconst:
                            // B1phase = B1freq * tSinceRF - phi1 + framePhase0
                            framePhase0 += -(B1freq - lastB1freq) * state.tSinceRF;
                            lastB1freq = state.B1freq;
                        });
                state.B1 = tmp1; state.B1 = tmp2; state.B1freq = tmp3;
            },
            cFolder++, createFromFolder--);

        guiGradientsFolder = cFolder; // folder index needed for updating during gradient pulses.
        guiAddFolder('Gradients: Gx=' + Math.round(state.Gx*100)/100 +
            ', Gy=' + Math.round(state.Gy*100)/100,   //there is no y unicode suffix. 
            'Gradients',
            function (guiFolder) {
                let tmp1 = state.Gx; let tmp2 = state.Gy;
                guiFolder.add(state, 'Gx', -7, 7, 1);
                guiFolder.add(state, 'Gy', -7, 7, 1);
                state.Gx = tmp1; state.Gy = tmp2;
            },
            cFolder++, createFromFolder--);

        guiAddFolder('Speed = ' + state.Gamma,
            'Speed',
            function (guiFolder) {
                let tmp1 = state.Gamma;
                guiFolder.add(state, 'Gamma', 0.1, 2.5, 0.5);
                state.Gamma = tmp1;
            },
            cFolder++, createFromFolder--);

        let FrameStr = (state.FrameStat ? 'Stationary' : '') + (state.FrameB0 ? 'B0' : '') +
            (state.FrameB1 ? 'B1' : '');

        guiAddFolder('Frame: ' + FrameStr,
            'Frame',
            function (guiFolder) {
                let tmp1 = state.FrameStat, tmp2 = state.FrameB0, tmp3 = state.FrameB1;
                guiFolder.add(state, 'FrameStat').name('Stationary').listen().
                    onChange(function () {
                        state.FrameStat = true;
                        state.FrameB0 = false;
                        state.FrameB1 = false;
                        floor.material = floorMaterial;
                        if (floorMaterial.visible)
                            shadowMaterialsInit(floorMaterial);
                    });
                guiFolder.add(state, 'FrameB0').name('B0').listen().
                    onChange(function () {
                        state.FrameStat = false;
                        state.FrameB0 = true;
                        state.FrameB1 = false;
                        let visible = floor.material.visible;
                        floor.material = frameFixed ?
                            floorMaterialFixed : floorMaterial;
                        if (visible)
                            shadowMaterialsInit(floor.material);
                    });
                guiFolder.add(state, 'FrameB1').name('B1').listen().
                    onChange(function () {
                        state.FrameStat = false;
                        state.FrameB0 = false;
                        state.FrameB1 = true;
                        let visible = floor.material.visible;
                        floor.material = frameFixed ?
                            floorMaterialFixed : floorMaterial;
                        if (visible)
                            shadowMaterialsInit(floor.material);
                    });
                state.FrameStat = tmp1; state.FrameB0 = tmp2; state.FrameB1 = tmp3;
            },
            cFolder++, createFromFolder--);

    } //guiInit

    function shadowMaterialsInit(floorMat) {

        // BasicMaterial is fast, but doesn't respond to light like Lambert, so colors
        // of materials don't match, even when color values do. 
        shadowMaterials.length = 0; //clears array of shadow colors.
        let material;
        for (let i = 0; i < nShadowColors; i++) {
            material = shadowMaterial.clone();//new THREE.MeshBasicMaterial();
            material.color.lerp(floorMat.color, i / nShadowColors * 2.8); //interpolate colors.
            // Last factor extrapolates colors for large i. Maybe this (alpha-blend?)
            // slows simulation down more than Lambert material shadows would, but
            // probably not since alpha-buffer is disabled. Current solution
            // works, but may break since lerp officially requires 0..1 arg.
            // Not pursued: Floor material could alternatively be changed
            // to basic to easily match colors. 
            shadowMaterials.push(material);
        }
        shadowMaterials[nShadowColors - 1].visible = false; //since extrapolation is not perfect.
    }

    function floorInit(geometry) {
        var floorGeo;
        switch (geometry) {
            case 'rect': floorGeo = new THREE.PlaneBufferGeometry(10, 10); break;
            case 'circle': floorGeo = new THREE.CircleBufferGeometry(6.5, 64); break;
        }
        var floor = new THREE.Mesh(floorGeo, floorMaterial);
        floor.position.z = -1.101;
        floor.receiveShadow = threeShadow;
        return floor;
    } //floorInit

    function B1init() {
        var cylMaterial = new THREE.MeshLambertMaterial({ color: "yellow" });
        B1cyl = cylinderMesh(new THREE.Vector3(0, 0, 0),
            new THREE.Vector3(0, 1, 0),
            cylMaterial, 8, radius);
        scene.add(B1cyl);
        if (myShadow) {
            // Shadows are initialized along y to make length right subsequently.
            B1shadow = shadowMesh(new THREE.Vector3(0, 1, 0));
            scene.add(B1shadow);
        }

    } //B1init

    function cameraInit() {
        camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 1, 10000);
        camera.up.set(0, 0, 1);
        camera.position.set(2.4, 5.6, 1.5); //probe camera.position for good coords.
        // Viewing angle/zoom control (no keybindings except panning):
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enablePan = false; //avoid using arrow keys (panning).
        controls.saveState();

        document.getElementById('ResetCamera').onclick = function () {
            controls.reset();
            document.getElementById('XYZview').innerHTML = 'XYZ';
        }

        document.getElementById('XYZview').onclick = function () {
            let current;
            switch (this.innerHTML) {
                case 'XYZ':
                case 'xyZ': current = 'Xyz';
                    camera.position.set(6, 0, 0); break;
                //controls.target is already (0,0,0), (focus 
                // and center of rotation), so no setting is needed.
                case 'Xyz': current = 'xYz';
                    camera.position.set(0, 6, 0);
                    break;
                case 'xYz': current = 'xyZ';
                    camera.position.set(0, -0.05, 6); //slightly off-axis to give shadow.
                    // Choice of off-axis is important to orient coordinate system logically.
                    // Increasing the off-axis viewing gives impression of off-center error.
                    break;
            }
            controls.update(); //needed whenever controls are changed manually.	
            this.innerHTML = current;
        };

    } //cameraInit

    function lightInit() {
        scene.add(new THREE.AmbientLight(0x707070));
        var light = new THREE.DirectionalLight(0xffffff, 1);
        light.castShadow = threeShadow;
        light.position.set(0, 0, 100);
        scene.add(light);
        scene.add(new THREE.DirectionalLightHelper(light, 0.2));
    }	//lightInit

    function text3dInit() {
        var loader = new THREE.FontLoader();
        // Local 'file:' use requires "chrome --allow-file-access-from-files"
        // loader.load( 'fonts/helvetiker_regular.typeface.json', function ( font ) {
        loader.load('https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', function (font) {
            var geometry = new THREE.TextBufferGeometry('Bloch', {
                font: font,
                size: 1,
                height: 0.1,
                curveSegments: 12,
                bevelEnabled: true,
                bevelThickness: 0.1,
                bevelSize: 0.1,
                bevelSegments: 5
            });
            var textMesh = new THREE.Mesh(geometry, shadowMaterial);
            textMesh.rotation.x = Math.PI / 2;
            textMesh.translateOnAxis(new THREE.Vector3(0, 0, -1), 3);
            scene.add(textMesh);
        });
    } //text3dInit

    function statsInit() {
        statsContainer = document.createElement('div');
        document.body.appendChild(statsContainer);
        stats = new Stats();
        statsContainer.appendChild(stats.dom);
    }

    function rendererInit() {

        let success = true;
        if (WEBGL.isWebGLAvailable()) {
            renderer = new THREE.WebGLRenderer({
                canvas: canvasA,
                antialias: true, //smooth edges.
                alpha: false
            }); //alpha may cost performance.
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = threeShadow;
            renderer.shadowMapSoft = true;

        } else {
            success = false;
            var warning = WEBGL.getWebGLErrorMessage();
            document.getElementById('WebGLmessage').appendChild(warning);
            dialog("dialogWebGLfail")();
            // renderer = new THREE.CanvasRenderer({ canvas: canvasA , antialias: true }); //CanvasRenderer is no longer part of THREE.
        }

        return success;
    } //rendererInit

    function ApplyToAll(matrix4) {
        for (var i = 0; i < state.IsocArr.length; i++) {
            state.IsocArr[i].M.applyMatrix4(matrix4); // result is Vector3
        };
    }

    function clearRepTimers() {
        
        isRunningSequence = false;
        window.clearTimeout(spoilTimer1);
        window.clearInterval(spoilTimer2);
        window.clearInterval(spoilTimer3);
        window.clearInterval(restartRepIfSampleChangeTimer);
        exciteTimers.forEach(function (item) { window.clearInterval(item) });
        exciteTimers = [];
        restartRepIfSampleChange = false;
        //clear sequence
        for (let i = 0; i <= eventCache.length; i++) {
            var id = eventCache[i];
            window.clearTimeout(id)
            window.clearInterval(id)
        }
        trigSampleChange = true;
        sampleChange();
        updateMenuList = [];
    }

    function exciteSpoilRepeat(TR, tipAngle, phaseCycle, spoiling, B1) {
        clearRepTimers();
        B1 = B1 || 4;

        let cycleLength = phaseCycle.length;
        for (let i = 0; i < cycleLength; i++) {
            window.setTimeout(function () {
                RFpulse('rect', tipAngle, phaseCycle[i], B1);
                exciteTimers.push(setInterval(function () {
                    RFpulse('rect', tipAngle, phaseCycle[i], B1)
                }, TR * cycleLength));
            }, i * TR);
        }

        if (spoiling) {
            let timeBeforeSpoil = TR - spoilDuration - 200
                - 300 * (state.Sample == 'Plane'); //reduction need for plane, especially.

            spoilTimer1 = window.setTimeout(function () { spoil() }, timeBeforeSpoil);
            spoilTimer2 = window.setTimeout(function () {
                spoilTimer3 = setInterval(function () {
                    spoil()
                }, TR)
            }, timeBeforeSpoil);
        }
    } // exciteSpoilRepeat


    function testFunction(TR, dict, B1) {
        // TR(ms) mean the duration of the sup puls
        // ~~TR can be change by the RF speed~~

        clearRepTimers();
        B1 = B1 || 4;
        let time = 0;
        let angleCache = [];

        // compute max value for normalization
        let maxAngle = 0;
        let maxAmp = 0;
        for (var id in dict)
        {
            var obj = dict[id]
            if (obj["RF"] != 0)
            {
                let rf = dict[id]["RF"];
                let ang = rf["angle"];
                maxAngle = Math.max(maxAngle, Math.max.apply(Math, ang));
            }
            if( obj["trap"]["Gx"]!=0)
            {
                let trap = dict[id]["trap"];
                let amp = trap["Gx"]["amplitude"];
                maxAmp = Math.max(maxAmp, Math.abs(amp));
            }
            if( obj["trap"]["Gy"]!=0)
            {
                let trap = dict[id]["trap"];
                let amp = trap["Gy"]["amplitude"];
                maxAmp = Math.max(maxAmp, Math.abs(amp));
            }
        }


        // loop the blocks
        for (var id in dict)
        {
            var obj = dict[id]
            if(obj["delay"] != 0) // global delay
            {
                var delayValue = obj["delay"] / state.Gamma;
                time += delayValue;
            }

            // compute the time separatly
            let t_rf = time;
            let t_adc = time;
            let t_gx = time;
            let t_gy = time;
            if(obj["RF"] != 0)
            {
                let rf = dict[id]["RF"];
                t_rf += rf["delay"] / state.Gamma; // add local delay here
                let ang = rf["angle"];
                let length = ang.length
                for (let i = 0; i < length; i++) // recreate the puls
                {
                     // reset the tSinceRF
                    state.tSinceRF = 0;
                    angleCache.push(ang[i])
                    eventCache.push(
                        window.setTimeout(function(){
                            isRunningSequence = false
                            if(state.FrameB0)
                            {
                                RFpulse('rect', Math.PI / 180 * ang[i],parseFloat(rf["phase"]) + 2 * Math.PI * rf["D_phase"][i], B1); // it's just work
                            }
                            else if(state.FrameB1)
                            {
                                RFpulse('rect', Math.PI / 180 * ang[i], parseFloat(rf["phase"]) + 2 * Math.PI * rf["D_phase"][i], B1); // it's just work
                            }
                            else
                            {
                                RFpulse('rect', Math.PI / 180 * ang[i], (-0.001 * 2 * TR* i + parseFloat(rf["phase"]))*state.Gamma + 2 * Math.PI * rf["D_phase"][i], B1);
                            }
                            GMvec.x =  ang[i] / maxAngle;
                            isRunningSequence = true;
                        }, t_rf)
                    );
                    t_rf += TR;
                }

                eventCache.push(
                    window.setTimeout(function(){
                        state.B1 = 0;
                    }, t_rf)
                );
            }
            if(obj["ADC"] != 0)
            {
                // G_ADC
                // todo Frequence and  phase
                let adc = dict[id]["ADC"];
                let period = adc["dwell"] * adc["num"] / state.Gamma;
                t_adc += adc["delay"] / state.Gamma;

                eventCache.push(
                    window.setTimeout(function(){
                        G_ADC.x = 1;
                    }, t_adc)
                );
                // ! adc time related with RF speed
                t_adc += period;

                eventCache.push(
                    window.setTimeout(function(){
                        G_ADC.x = 0;
                    }, t_adc)
                );
            }
            if( obj["trap"]["Gx"]!=0)
            {
                let trap = dict[id]["trap"];
                let amp = trap["Gx"]["amplitude"];

                t_gx += trap["Gx"]["delay"] / state.Gamma;;
                let repeatTime = 10;
                let tempTime = Math.ceil(trap["Gx"]["period"] / state.Gamma /repeatTime);
                for (let i = 0; i < repeatTime; i++)
                {
                    t_gx += tempTime
                    eventCache.push(
                        window.setTimeout(function(){
                            gradPulse(amp / repeatTime);
                            GMvec.y =  amp/maxAmp
                        }, t_gx )
                    );
                }
            }
            if( obj["trap"]["Gy"]!=0)
            {
                let trap = dict[id]["trap"];
                let amp = trap["Gy"]["amplitude"]

                t_gy += trap["Gy"]["delay"] / state.Gamma;

                let repeatTime = 10;
                let tempTime = Math.ceil(trap["Gy"]["period"] / state.Gamma /repeatTime)
                for (let i = 0; i < repeatTime; i++)
                {
                    t_gy += tempTime

                    eventCache.push(
                        window.setTimeout(function(){
                            gradPulse(amp  / repeatTime, Math.PI / 2);
                            GMvec.z =   amp/maxAmp
                        }, t_gy )
                    );
                }
            }

            eventCache.push(
                window.setTimeout(function()
                {
                    GMvec.x = 0 // rf
                    isRunningSequence = false; // rf
                }, t_rf)
            );
            eventCache.push(
                window.setTimeout(function()
                {
                    G_ADC.x = 0;// adc
                    fidbox.style["backgroundColor"] = "transparent"; // adc
                }, t_adc)
            );

            eventCache.push(
                window.setTimeout(function()
                {
                    GMvec.y = 0
                }, t_gx)
            );
            eventCache.push(
                window.setTimeout(function()
                {
                    GMvec.z = 0
                }, t_gy)
            );
            time = Math.max(t_rf, t_gx, t_gy,t_adc);
        }
    } // testFunction


    function buttonAction(label) {

        if (paused) {
            paused = false;
            $("#Pause").button("option", "label", "||");
            //$( "#Pause" ).button( "option", "label", "⏸");}
        }
        let TR;
        switch (label) {
            //  add some thing zhaoshun
            case "Load_seq_file":
                loadSeq().then(function(d){
                    let seq = readString(d);
                    var seqDict = seq.getSeq(state.Gamma);
                    console.log(seqDict);
                    testFunction(50, seqDict, 512)
                });
                break;
            case "Precession": state.Sample = "Precession";
                trigSampleChange = true; break;
            case "Equilibrium": state.Sample = "Equilibrium";
                trigSampleChange = true; break;
            case "Inhomogeneity": state.Sample = "Inhomogeneity";
                trigSampleChange = true; break;
            case "Mixed matter": state.Sample = "Mixed matter";
                trigSampleChange = true; break;
            case "Weak gradient": state.Sample = "Weak gradient";
                trigSampleChange = true; break;
            case "Strong gradient": state.Sample = "Strong gradient";
                trigSampleChange = true; break;
            case "Structure": state.Sample = "Structure";
                trigSampleChange = true; break;
            case "Ensemble": state.Sample = "Ensemble";
                trigSampleChange = true; break;
            case "Plane": state.Sample = "Plane";
                trigSampleChange = true; break;
            //	    case "90°ʸ hard" : ApplyToAll(propagator90y); break;
            case "90°ₓ hard": RFpulse('rect', Math.PI / 18 * 9,  Math.PI,       4); break;
            case "90°ʸ hard": RFpulse('rect', Math.PI / 18 * 9, -Math.PI / 2,   4); break;
            case "80°ₓ hard": RFpulse('rect', Math.PI / 18 * 8,  Math.PI,       4); break;
            case "30°ₓ hard": RFpulse('rect', Math.PI / 18 * 6,  Math.PI,       4); break;
            case "30°ʸ hard": RFpulse('rect', Math.PI / 18 * 6, -Math.PI / 2,   4); break;
            case "90°ₓ sinch": RFpulse('sinc',Math.PI / 18 * 9,  Math.PI,       4); break;

            case "90°ₓ soft": RFpulse('rect', Math.PI / 18 * 9,  Math.PI,       0.3); break;
            case "90°ʸ soft": RFpulse('rect', Math.PI / 18 * 9, -Math.PI / 2,   0.3); break;
            case "30°ₓ soft": RFpulse('rect', Math.PI / 18 * 3,  Math.PI,       0.3); break;
            case "30°ʸ soft": RFpulse('rect', Math.PI / 18 * 3, -Math.PI / 2,   0.3); break;
            case "90°ₓ sincs": RFpulse('sinc',Math.PI / 18 * 9,  Math.PI,       0.8); break;

            case "180°ʸ": RFpulse('rect', Math.PI / 18 * 18, -Math.PI / 2,  8); break;
            case "180°ₓ": RFpulse('rect', Math.PI / 18 * 18,  Math.PI,      8); break;
            case "160°ʸ": RFpulse('rect', Math.PI / 18 * 16, -Math.PI / 2,  8); break;
            case "160°ₓ": RFpulse('rect', Math.PI / 18 * 16,  Math.PI,      8); break;
            case "180°ʸ sincs": RFpulse('sinc', Math.PI, -Math.PI / 2, 1.6); break;

            case "Spoil": $("#Presets").css('color', '#bbbbbb');
                spoil(); break;

            case "Gx refocus": $("#Presets").css('color', '#bbbbbb');
                if (frameFixed) gradRefocus(); break;

            case "Gx pulse": $("#Presets").css('color', '#bbbbbb');
                if (frameFixed) gradPulse(2); break; //whatever area

            case "Gy pulse": $("#Presets").css('color', '#bbbbbb');
                if (frameFixed) gradPulse(2, Math.PI / 2); break; //whatever area

            case "Non-rep. exc.":
                GMvec = [0,0,0]
                G_ADC = [0,0,0]
                clearRepTimers();
                break;

            case "[90°ₓ] TR=5s,spoiled":
                TR = 5000; //ms
                exciteSpoilRepeat(TR, Math.PI / 2, [Math.PI], true);
                break;

            case "[30°ʸ] TR=3s,spoiled":
                TR = 3000; //ms
                exciteSpoilRepeat(TR, Math.PI / 6, [-Math.PI / 2], true);
                break;

            case "[90°ʸ] TR=5s,spoiled":
                TR = 5000; //ms
                exciteSpoilRepeat(TR, Math.PI / 2, [-Math.PI / 2], true);
                break;

            case "[90°ʸ] TR=8s,spoiled":
                TR = 8000; //ms
                exciteSpoilRepeat(TR, Math.PI / 2, [-Math.PI / 2], true);
                break;

            case "[90°ₓ] TR=5s":
                TR = 5000; //ms
                exciteSpoilRepeat(TR, Math.PI / 2, [Math.PI], false);
                break;

            case "[±90°ₓ] TR=5s":
                TR = 5000; //ms
                exciteSpoilRepeat(TR, Math.PI / 2, [Math.PI, 0], false);
                break;

            case "90°ₓ-[180°ʸ]ES=5s":
                let ES = 5000; //ms
                RFpulse('rect', Math.PI / 2, Math.PI, 4);
                window.setTimeout(
                    function () {
                        exciteSpoilRepeat(ES, Math.PI, [-Math.PI / 2], false, 8);
                        restartRepIfSampleChange = true; //cleared in clearRepTimers()
                    },
                    ES / 2);
                restartRepIfSampleChange = true; // Restart cycle upon sample change.
                break;
            case "||":
            case "⏸":
            case "\u25AE\u25AE":
                paused = true;
                $("#Pause").button("option", "label", "▶");
                break;
            case "▶":
                break; // Any button restarts so corresponding action appears above.

            case "Save":
                savedState = Object.assign({}, state);
                delete (savedState.IsocArr);
                savedState2 = [];
                state.IsocArr.forEach(function (item) {
                    savedState2.push({
                        Mx: item.M.x,
                        My: item.M.y,
                        Mz: item.M.z
                    })
                });
                $("#Presets").button("option", "label", "Saved");
                savedFlag = true;  // Prevent sample changes until scene change.
                guiFolderFlags = [true, true, true, true, true, true, true, true, true]; //close all
                break;
            case "Saved":
                Object.assign(state, savedState);
                state.IsocArr.forEach(function (item, index) {
                    (index >= savedState2.length) && console.log('shouldnt happen');
                    let saved = savedState2[index];
                    item.M.x = saved.Mx;
                    item.M.y = saved.My;
                    item.M.z = saved.Mz;
                });
                updateMenuList.push(2); // trigger dat-gui update of all but first folders.
                break;
            default: alert("Button with no action pressed: " + label);
        }
        if (trigSampleChange) {
            if (hideWhenSelected) {
                let menuItem = label.replace(/\s/g, ''); //no space in ids.
                $("#" + menuItem).hide(); // hide menu option on button.
            }
            savedFlag = false; //restoring saved state is no longer possible.
        }

    } // buttonAction

    function addConfigButton(id, leftid) {
        $("#" + id)
            .button()
            .css({ 'padding-left': '0em', 'padding-right': '0em' }) //width is explicit in html
            .click(function () {  //add functionality for left part
                var label = this.textContent || this.innerText || ""; //fallbacks for limited browsersupport.
                if ((this.id != "Presets") || (label == "Saved") || reloadSceneResetsParms) {
                    buttonAction(label);
                } else {
                    trigSampleChange = true;
                }
            })
            .next() // format right part
            .button({
                text: false,
                icons: {
                    primary: "ui-icon-triangle-1-s"
                }
            })
            .click(function () { //add functionality for right part
                let submenu = $(this).parent().next();
                let submenuOpen = submenu.is(":visible");
                $(".DropDowns:visible").hide(); //close all submenus
                if (!submenuOpen) //dont reopen if user is trying to close.
                    submenu.show().position({
                        of: $("#" + id), //target element (reference)
                        my: "left bottom",  //placed object ref point
                        at: "left top", //target element ref point
                        collision: "none"
                    });

                $('html').one("click", function () { //hide menu if click outside of doc. Run once.
                    submenu.hide();
                });


                return false;
            })
            .parent()
            .position({
                of: $("#" + leftid), my: "left top", at: "right+10 top",
                collision: "flip"
            }) //place right of leftid dropdown button.
            .controlgroup()  // groups the two half buttons
            .next()  // go to next node
            .hide()  //hide menu
            .menu();  //make right part a menu

        $("." + id + "Action").click(function () {
            if (hideWhenSelected && (id == "Presets")) {
                //show the hidden Presets button menu option.
                let currButtonLabel = $("#Presets").text();
                currButtonLabel = currButtonLabel.replace(/\s/g, ''); //no space in ids.
                $("#" + currButtonLabel).show();
            }
            $("#" + id).button("option", "label", this.innerHTML); //sets the option "label".
            buttonAction(this.textContent || this.innerText);
        });
    } // addConfigButton

    function closeMenuIfOpened(id) {
        document.getElementById(id).onclick = function () {
            guiFolderFlags.forEach(function (item, index) {
                (!item) && guiFolders[index].close();
            })
        }
    } // closeMenuIfOpened


    function addButton(id, leftid) {
        $("#" + id)
            .button()
            .click(function () {  //add functionality for left part
                var label = this.textContent || this.innerText || ""; //fallbacks for limited browsersupport.
                buttonAction(label);
            })
            .parent()
            .position({ of: $("#" + leftid), my: "left top", at: "right+10 top" }); //place right of leftid dropdown button.

        $("." + id + "Action").click(function () {  // adds action to buttons in the submenu.
            $("#" + id).button("option", "label", $(this).text()); //sets the option "label".
        });
    } // addButton

    function init() {

        scene = new THREE.Scene();

        if (myShadow) { //central dot once
            var originShadowGeo = new THREE.CircleBufferGeometry(radius, 8, 0, 2 * Math.PI);
            // dot displacement error comes from z-view being off-axis for visibility.
            var originShadow = new THREE.Mesh(originShadowGeo, shadowMaterial);
            originShadow.position.z = -1.099;
            scene.add(originShadow);
        }


        doStats && statsInit();
        magInit();
        guiInit();
        shadowMaterialsInit(floorMaterial);
        floorCirc = floorInit('circle');
        floorRect = floorInit('rect');
        floor = floorRect;
        scene.add(floor);

        state.Sample = "Precession";
        trigSampleChange = true;

        initFIDctxAxis();
        initGMctxAxis(); // update 07.2022
        // text3dInit(); //works. Maybe use for startup.
        B1init();
        lightInit();
        if (!rendererInit()) return;
        cameraInit();
        addConfigButton("Presets", "leftmost"); //Add button with id "Preset" right of "leftmost"
        addConfigButton("ExcHard", "PresetsDrop");
        addConfigButton("Soft", "ExcHardDrop");
        addConfigButton("Refocus", "SoftDrop");
        addConfigButton("Spoil", "RefocusDrop");
        addConfigButton("RepExc", "SpoilDrop");
        addButton("Pause", "RepExcDrop");
        closeMenuIfOpened("PresetsDrop");
        closeMenuIfOpened("ExcHardDrop");
        closeMenuIfOpened("SoftDrop");
        adjustToScreen();
        window.setTimeout(0); //flush cache;
        adjustToScreen(); //calling twice improves view.

        $("#newBlochSimulator").dialog({
            modal: false,
            buttons: {
                'Get help': function () {
                    let guiFolder = guiFolders[0];
                    guiFolder.open();
                    guiFolderFlags[0] = true;
                    // dialog("dialogGetStarted"); Doesnt work. Not worth trying.
                    // $( "#dialogGetStarted" ).show();//dialog( "open" );
                },
                'Proceed': function () {
                    $(this).dialog("close")
                }
            }
        });

        let elem = document.getElementById('FloorShape');
        let next, material;
        elem.onclick = function () {
            scene.remove(floor);
            switch (elem.innerHTML) {
                case '⬛': next = '◯';
                    floor.material.visible = false;
                    material = floorMaterialBlack;
                    shadowMaterialsInit(material);
                    break;
                case '⬜': next = '⬛';
                    material = floor.material; //remember material
                    material.visible = true;
                    floorRect.material = material;
                    scene.add(floor = floorRect);
                    break;
                case '◯': next = '⬜';
                    material = floor.material; //remember material
                    material.visible = true;
                    floorCirc.material = material;
                    scene.add(floor = floorCirc);
                    break;
            }
            shadowMaterialsInit(material);
            elem.innerHTML = next;

        }

        if (addAxisHelper) {
            var axisHelper = new THREE.AxisHelper(3); //length. Shows x,y,z. Works
            scene.add(axisHelper);
        }

        document.body.appendChild(renderer.domElement);

        //	  THREEx.FullScreen.bindKey({ charCode : 'm'.charCodeAt(0) }); //Requires game extension.

    } //init

    function guiUpdate() {
        // Separate guiUpdate loop in case framerate is low:
        //	    window.setTimeout(guiUpdate, 100); //ms
        nFolder = guiFolders.length;
        let updateFolder = updateMenuList.shift(); //Folder marked for updating? Else undefined.

        for (var cFolder = 0; //look for changes in open/closed status
            guiFolderFlags[cFolder] == guiFolders[cFolder].closed; //As long as saved close-status matches actual.
            cFolder++) {
            if ((updateFolder) && (updateFolder == cFolder)) { ;break }; // Folder marked for update?
            if (cFolder == nFolder - 1) { //Check if finished. Loop terminates before if change.
                if (trigSampleChange) {
                    sampleChange();
                    debug && console.log('guiUpdate ' + state.Sample);
                    guiFolderFlags = [true, true, true, true, true, true, true, true, true]; //close all
                    guiInit(guiFolderStrs[1]); // Reinitialize GUI
                };
                return;
            } //nothing changed except possibly preset.
        }
        // A folder change was found:
        $("#Presets").css('color', '#bbbbbb'); //dim Scene button.
        if (!updateFolder) //only toggle open/close if update was triggered by click.
            guiFolderFlags[cFolder] = !(guiFolderFlags[cFolder]);
        for (let cFolder2 = cFolder + 1; cFolder2 < nFolder; cFolder2++) { //close all following open folders.
            guiFolderFlags[cFolder2] = true;
        }
        state.B1 = Math.round(state.B1 * 10) / 10; // occasionally B1 ends up with many digits.
        guiInit(guiFolderStrs[cFolder]); //Re-initialize folders following the changed one.
    } //guiUpdate

    function relaxThermal() {
        let nIsoc = state.IsocArr.length;
        let rep, Mx, My, Mz, Mxy, arg, randomIsocIndi, randomIsoc, R1;
        if (state.T1 != Infinity) {
            R1 = 1 / (state.T1 + 0.1); //precision is not an aim here.
            for (rep = 1; rep < Math.floor(nIsoc * R1 / 10); rep++) { //TODO: frame rate needs to enter.
                //repeat depending on T1 and nIsoc
                Mz = thermalDrawFromLinearDist(state.B0); //cosTheta is linearly distributed.
                Mxy = Math.sqrt(1 - Mz * Mz);
                randomIsocIndi = Math.floor(nIsoc * Math.random());
                randomIsoc = state.IsocArr[randomIsocIndi];
                arg = Math.random() * 2 * Math.PI;
                randomIsoc.M.fromArray([Mxy * Math.cos(arg), Mxy * Math.sin(arg), Mz]);
            }
        }
        else
            R1 = 0;
        // Additional T2 relaxation, if needed:
        let R2 = 1 / (state.T2 + 0.1); //precision is not an aim here.
        if (state.T2 != Infinity) {
            for (rep = 1; rep < Math.floor(nIsoc * (R2 - R1) / 10); rep++) { //TODO: frame rate needs to enter.
                randomIsocIndi = Math.floor(nIsoc * Math.random());
                randomIsoc = state.IsocArr[randomIsocIndi];
                Mx = randomIsoc.M.x;
                My = randomIsoc.M.y;
                Mxy = Math.sqrt(Mx * Mx + My * My);
                arg = Math.random() * 2 * Math.PI;
                randomIsoc.M.x = Mxy * Math.cos(arg);
                randomIsoc.M.y = Mxy * Math.sin(arg);
            }
        }
    } //relaxThermal

    function BlochStep(dt) {
        state.t += dt;
        state.tSinceRF += dt;

        let gamma = state.Gamma;
        let B0, B1freq;

        if ((frameFixed) && (!state.FrameStat)) { //reduce B0 and B1freq if frame is fixed.
            if (state.FrameB1) {//B1
                B0 = state.B0 - state.B1freq / gamma;
                B1freq = 0;
            }
            else
                if (state.FrameB0) { // B0
                    B0 = 0;
                    B1freq = state.B1freq - state.B0 / gamma;
                }
        }
        else {//stationary
            B0 = state.B0;
            B1freq = state.B1freq;
        }


        let Gx = state.Gx;
        let Gy = state.Gy;
        if (state.areaLeftGrad != 0) { // is gradient refocusing ongoing?
            let angle = state.PulseGradDirection;
            let dArea = dt * gamma * (Math.cos(angle) * Gx + Math.sin(angle) * Gy);
            if (Math.abs(dArea) < Math.abs(state.areaLeftGrad)) //mid grad pulse?
                state.areaLeftGrad -= dArea;
            else { // last time step
                Gx *= state.areaLeftGrad / dArea * Math.cos(angle);
                Gy *= state.areaLeftGrad / dArea * Math.sin(angle);
                state.areaLeftGrad = 0;
                state.Gx = 0;
                state.Gy = 0;
                updateMenuList.push(guiGradientsFolder); //mark gradient folder for updating
            }
        }


        let B1 = state.B1;
        let B1vec, envelope;
        if (B1 == 0) {
            B1vec = nullvec.clone();
        }
        else
            if (state.tLeftRF >= 0) { // pulsing.
                // [B1vec, envelope] = state.RFfunc(B1, B1freq); //not IE compatible
                let retval = state.RFfunc(B1, B1freq);
                B1vec = retval[0];
                envelope = retval[1];
                let dArea = dt * gamma * envelope;
                if (state.tLeftRF < dt) { //end of RF pulse
                    // Adjust B1 to match tip angle on resonance (by adding delta pulse):
                    B1vec.multiplyScalar(state.areaLeftRF / dArea);
                    state.areaLeftRF = 0; // end pulse
                    state.tLeftRF = 0;
                    state.B1 = 0;
                    updateMenuList.push(guiFieldsFolder); //mark field folder for updating
		            delayB1vecUpdate = 5; // delay 1 frame to avoid corrected B1 to be shown briefly.
                } else { //mid pulse
                    state.areaLeftRF -= dArea;
                    state.tLeftRF -= dt;
                }
            } else { //non-pulsed RF
                // [B1vec, envelope] = state.RFfunc(B1, B1freq); //not IE compatible
                let retval = state.RFfunc(B1, B1freq);
                B1vec = retval[0];
                envelope = retval[1];
            }

        let f1, f2, RelaxFlag, isoc;

        if (state.Sample == 'Thermal ensemble') {
            relaxThermal();
            RelaxFlag = false;
        } else {
            if ((state.T1 == Infinity) && (state.T2 == Infinity)) {
                RelaxFlag = (spoilR2 != 0);
                f1 = 1.; if (RelaxFlag) { f2 = Math.exp(-dt * spoilR2) };
            } else {
                f1 = Math.exp(-dt / state.T1); f2 = Math.exp(-dt * (1. / state.T2 + spoilR2));
                RelaxFlag = true;
            }
        }

        for (let Cspin = 0; Cspin < state.IsocArr.length; Cspin++) {
            isoc = state.IsocArr[Cspin];
            isoc.detuning = isoc.dB0 + (Gx * isoc.pos.x + Gy * isoc.pos.y) / gradScale;
            let Bvec = B1vec.clone().addScaledVector(unitZvec, B0 + isoc.detuning);
            isoc.detuning = (B0 + isoc.detuning) / gamma - B1freq;

            let Bmag = Bvec.length();
            if (Bmag != 0) {
                isoc.M.applyAxisAngle(Bvec.divideScalar(Bmag),
                    -Bmag * dt * gamma);
            }

            if (!B1vec.equals(nullvec)) {
                isoc.dMRF.crossVectors(isoc.M, B1vec).multiplyScalar(gamma);//torque
            } else
                isoc.dMRF = nullvec.clone();

            if (RelaxFlag) {
                let df2 = isoc.dR2 ? Math.exp(-isoc.dR2 * dt) : 1; //extra relax for isoc
                let df1 = (isoc.dR1 && ((state.T1 * state.T2) != Infinity) && (spoilR2 == 0)) ?
                    Math.exp(-isoc.dR1 * dt) : 1;
                let M0 = isoc.M0 ? isoc.M0 : 1;
                isoc.M.set(isoc.M.x * f2 * df2,
                    isoc.M.y * f2 * df2,
                    isoc.M.z * f1 * df1 + (1. - f1 * df1) * M0);
            }

            state.IsocArr[Cspin] = isoc;
        }

        // Debugging: Check if nullvec ever changes.
        if (debug && (!nullvec.equals(new THREE.Vector3(0., 0., 0.)))) {
            alert("nullvec changed!")
        }

        return B1vec;

    } //BlochStep

    function updateFid(sample, FIDtimes, FIDvalues, color, view) {
        if (isNaN(sample)) return;
        FIDvalues.push(sample);
        FIDtimes.push(state.t);
        let FIDdurSecs = FIDduration / 1000;
        while (FIDtimes[0] < (state.t - FIDdurSecs)) {
            FIDtimes.shift(); // Don't merge with value check.
            FIDvalues.shift();
        }
        if (!view) { return }

        FIDctx.save();
        FIDctx.strokeStyle = color;
        FIDctx.lineWidth = (color == blueStr) ? 3 : 2;
        FIDctx.translate(0, Math.floor(grHeight / 2));
        FIDctx.beginPath();
        let len = FIDvalues.length;
        let FidEnd = FIDtimes[len - 1];
        let downSample = Math.floor(len / 200) + 1; // downsample above 200 points.

        if (downSample == 1) // no down-sampling
            FIDvalues.forEach(
                function (item, index) { //first lineTo is interpreted as moveTo
                    FIDctx.lineTo( //rounding of the values cost smoothness.
                        (1 - (FidEnd - FIDtimes[index]) / FIDdurSecs) * grWidth,
                        -item * grHeight / 2);
                })
        else { // Skip some line elements to save time.
            let FIDtimesDownSampled = FIDtimes.filter((e, i) => (i % downSample == 0));
            FIDvalues.filter((e, i) => (i % downSample == 0)).forEach( // keep every downSample'th
                function (item, index) { //first lineTo is interpreted as moveTo
                    FIDctx.lineTo( //rounding of the values cost smoothness.
                        (1 - (FidEnd - FIDtimesDownSampled[index]) / FIDdurSecs) * grWidth,
                        -item * grHeight / 2);
                });
        }
        FIDctx.stroke();
        FIDctx.restore();
    } //updateFidWrap

    function updateGM(sample, GMtimes, GMvalues, color, view) {
        if (isNaN(sample)) return;
        GMvalues.push(sample);
        GMtimes.push(state.t);
        let GMdurSecs = FIDduration / 1000;
        while (GMtimes[0] < (state.t - GMdurSecs)) {
            GMtimes.shift(); // Don't merge with value check.
            GMvalues.shift();
        }
        if (!view) { return }

        GMctx.save();
        GMctx.strokeStyle = color;
        GMctx.lineWidth = (color == blueStr) ? 3 : 2;
        GMctx.translate(0, Math.floor(grHeight / 2));
        GMctx.beginPath();
        let len = GMvalues.length;
        let GMEnd = GMtimes[len - 1];
        let downSample = Math.floor(len / 200) + 1; // downsample above 200 points.
        if (downSample == 1) // no down-sampling
            GMvalues.forEach(
                function (item, index) { //first lineTo is interpreted as moveTo
                    GMctx.lineTo( //rounding of the values cost smoothness.
                        (1 - (GMEnd - GMtimes[index]) / GMdurSecs) * grWidth,
                        -item * grHeight / 2);
                })
        else { // Skip some line elements to save time.
            let GMtimesDownSampled = GMtimes.filter((e, i) => (i % downSample == 0));
            GMvalues.filter((e, i) => (i % downSample == 0)).forEach( // keep every downSample'th
                function (item, index) { //first lineTo is interpreted as moveTo
                    GMctx.lineTo( //rounding of the values cost smoothness.
                        (1 - (GMEnd - GMtimesDownSampled[index]) / GMdurSecs) * grWidth,
                        -item * grHeight / 2);
                });
        }
        GMctx.stroke();
        GMctx.restore();
    } //updateGM

    function updateFidWrap(Mx, Mz, Mxy, color) {
        switch (color) {
            case white:
                updateFid(Mx, MxTimes, MxCurve, 'red', state.viewMx);
                updateFid(Mz, MzTimes, MzCurve, 'gray', state.viewMz);
                updateFid(Mxy, MxyTimes, MxyCurve, 'white', state.viewMxy);
                break;
            case green:
                updateFid(Mxy, curveGreenTimes, curveGreen, greenStr, state.viewMxy);
                break;
            case blue:
                updateFid(Mxy, curveBlueTimes, curveBlue, blueStr, state.viewMxy);
                break;
            default: alert("color should be specified");
        }

    } //updateFidWrap

    function updateGMWrap(v1, v2, v3, mode) {
        switch (mode) {
            case "RFandGM":
                updateGM(v2, GxTimes, GxCurve, 'green', state.viewGx);
                updateGM(v3, GyTimes, GyCurve, 'yellow', state.viewGy);
                updateGM(v1, RFTimes, RFCurve, 'white', state.viewRF);
                break;
            case "ADC":
                updateGM(v1, GadcTimes, GadcCurve, 'red', state.viewRF);
                // updateGM(Gy, GxTimes, GxCurve, 'gray', state.viewGx);
                // updateGM(RF, GyTimes, GyCurve, 'white', state.viewGy);
                break;
        // default: alert("color should be specified");
        }

    } //updateGMWrap

    function animate(time) {
        requestAnimationFrame(animate);
        elapsed = time - then;
        if (elapsed < fpsInterval) { return } // skip frames with long delays (loss of focus).
        then = time - (elapsed % fpsInterval);
        dt = (time - lastTime) / 1000;
        lastTime = time;
        if (dt > 0.1) { return };
        guiTimeSinceUpdate += dt;
        if (guiTimeSinceUpdate > guiUpdateInterval) {
            guiUpdate();
            guiTimeSinceUpdate = 0;
        }

        if (!paused) {

            // Stats on dt:
            dtTotal += dt; dtCount++;
            dtMemory[dtMemIndi] = dt; //short term memory of dt 
            dtMemIndi = (dtMemIndi + 1) % dtMemory.length;

            let B1vec = BlochStep(dt);
            let B1mag = B1vec.length();
            let allScale = state.allScale;

            let viewingAngle = controls.getPolarAngle(); //floor is lowered so Math.PI/2 isnt a thresh.
            let downViewRatio = 1 - Math.min(viewingAngle / downViewThresh, 1); // 1..0
            let shadowMaterial = shadowMaterials[Math.round(downViewRatio * (nShadowColors - 1))];
            // Clear FID
            FIDctx.clearRect(-5, -5, grWidth + 10, grHeight + 10); //asym borders are needed
            GMctx.clearRect(-5, -5, grWidth + 10, grHeight + 10); //asym borders are needed

            if ((B1mag != 0) && state.viewB1)
            { // view B1
                B1cyl.quaternion.
                    setFromUnitVectors(unitYvec, B1vec.clone().divideScalar(B1mag));
                B1cyl.scale.y = B1mag * B1scale * allScale;
                B1cyl.visible = true;

                if (myShadow) {  //shadow of B1
                    var B1vecTrans = B1vec.clone().projectOnPlane(unitZvec);
                    var B1vecTransLength = B1vecTrans.length();

                    B1shadow.material = shadowMaterial;
                    B1shadow.quaternion.
                                    setFromUnitVectors(unitYvec, B1vecTrans.clone().divideScalar(B1vecTransLength));
                    B1shadow.scale.y = B1vecTransLength * B1scale * allScale;  // requires length_y=1 initially.
                    B1shadow.visible = true;
                }
            }
            else{
                if ((delayB1vecUpdate--) <= 0) { //used to delay updating of B1-viewing to prevent flickering.
                    delayB1vecUpdate = 0; 
                    // console.log(delayB1vecUpdate);
                    B1cyl.visible = false;
                    B1shadow.visible = false;
                }
            }

            let Mvec, dMRFvec, torqueStart, isoc;
            let Mtot = nullvec.clone();
            let nIsoc = state.IsocArr.length;
            let showTotalCurve = true;

            let Gtot = nullvec.clone();
            let Atot = nullvec.clone();
            for (let i = 0; i < nIsoc; i++) {
                isoc = state.IsocArr[i];

                Mvec = isoc.M;

                Mtot.add(Mvec);
                Gtot.add(GMvec);
                Atot.add(G_ADC);
                dMRFvec = isoc.dMRF; //TODO: consistent naming for vectors

                // View effective B1 and shadow:
                isoc.B1eff.visible = false;
                if (!isRunningSequence)
                    isoc.tshadow.visible = false;
                if ((state.FrameB1) && (state.viewTorqB1eff) && // View B1eff if B1-frame is chosen.
                    // Hmm, the following line was probably introduced for a good reason (maybe Ensemble), but implies that only off-resonance B1eff is shown for single isochromate if off-resonance. I'll add "|| (i==0)" as test (note that the test starts above):
                    //			((isoc.detuning != 0 ) || frameFixed)) { // Only show on-res B1eff when off-center isocs are present.
                    ((isoc.detuning != 0) || frameFixed || (i == 0))) { // Only show on-res B1eff when off-center isocs are present.
                    let B1eff = B1vec.clone().addScaledVector(unitZvec, isoc.detuning);
                    let B1effMag = B1eff.length(); //TODO: consistent naming for magnitudes
                    if (B1effMag != 0) {
                        isoc.B1eff.visible = true;
                        isoc.B1eff.quaternion.
                            setFromUnitVectors(unitYvec, B1eff.clone().divideScalar(B1effMag));
                        isoc.B1eff.scale.y = B1effMag * B1scale * allScale;
                        isoc.B1eff.visible = true;
                        isoc.B1eff.position.set(isoc.pos.x, isoc.pos.y, isoc.pos.z);

                        if (myShadow) {  //shadow of B1eff replaces torque shadow, if B1eff is shown.
                            var B1effTrans = B1eff.clone().projectOnPlane(unitZvec);
                            var B1effTransLength = B1effTrans.length();
                            isoc.tshadow.material = shadowMaterial;
                            isoc.tshadow.quaternion.
                                setFromUnitVectors(unitYvec, B1effTrans.clone().divideScalar(B1effTransLength));
                            isoc.tshadow.scale.y = B1effTransLength * B1scale * allScale;  // requires length_y=1 initially.
                            isoc.tshadow.visible = true;
                        }
                        else
                            if (!isRunningSequence)
                                isoc.tshadow.visible = false;
                    }
                }

                var MvecLength = Mvec.length();
                if (MvecLength > 0.005) { // Show magnetization
                    // Sets quaternion to rotate direction vector One to direction vector Two:
                    isoc.cylMesh.quaternion.
                        setFromUnitVectors(unitYvec, Mvec.clone().divideScalar(MvecLength));
                    isoc.cylMesh.scale.y = MvecLength * allScale; // requires length_y=1 initially.
                    isoc.cylMesh.position.set(isoc.pos.x, isoc.pos.y, isoc.pos.z);
                    torqueStart = Mvec.clone().clampLength(0, Mvec.length() * allScale - radius / 2).add(isoc.pos);
                    isoc.torque.position.set(torqueStart.x, torqueStart.y, torqueStart.z);
                    isoc.cylMesh.visible = true;
                    isoc.shadow.visible = true;
                }
                else
                    isoc.cylMesh.visible = false;

                var MvecTrans = Mvec.clone().projectOnPlane(unitZvec).multiplyScalar(allScale);
                var MvecTransLength = MvecTrans.length();
                if (myShadow) {  //shadow of magnetization
                    if (MvecTransLength > 0.005) {

                        isoc.shadow.material = shadowMaterial;

                        isoc.shadow.quaternion.
                            setFromUnitVectors(unitYvec, MvecTrans.clone().divideScalar(MvecTransLength));
                        isoc.shadow.position.set(isoc.pos.x, isoc.pos.y, 0);
                        isoc.tshadow.position.set(isoc.pos.x, isoc.pos.y, 0); //if tshadow is used for B1eff. Else overwritten below.
                        isoc.shadow.scale.y = MvecTransLength;
                    } // requires length_y=1 initially.
                    else
                        isoc.shadow.visible = false;
                }

                var dMRFvecLength = dMRFvec.length();
                if (((dMRFvecLength < 0.01) || (!state.viewTorqB1eff)) || (state.FrameB1)) { // Show torque except in B1-frame. //
                    if (!isRunningSequence)
                        isoc.torque.visible = false;
                    
                }
                else { // draw torque
                    isoc.torque.visible = true;
                    isoc.tshadow.visible = true;
                    isoc.torque.quaternion.
                        setFromUnitVectors(unitYvec, dMRFvec.clone().divideScalar(dMRFvecLength));
                    isoc.torque.scale.y = dMRFvecLength * torqueScale * allScale; // requires length_y=1 initially.

                    if (myShadow) { // draw torque shadow
                        var dMRFvecTrans = dMRFvec.clone().projectOnPlane(unitZvec);
                        var dMRFvecTransLength = dMRFvecTrans.length();

                        if (dMRFvecTransLength > 0.005) {
                            isoc.tshadow.material = shadowMaterial;
                            isoc.tshadow.quaternion.
                                setFromUnitVectors(unitYvec, dMRFvecTrans.clone().divideScalar(dMRFvecTransLength));
                            isoc.tshadow.position.set(isoc.pos.x + MvecTrans.x, isoc.pos.y + MvecTrans.y, 0);
                            isoc.tshadow.scale.y = dMRFvecTransLength * torqueScale * allScale;
                        }// requires length_y=1 initially.
                        else
                        
                            if (!isRunningSequence)
                                isoc.tshadow.visible = false;
                    }
                    else
                        isoc.tshadow.visible = false;
                }

                if (isoc.showCurve) { // Update Mxy curve for this isochromate.
                    updateFidWrap(Mvec.x, Mvec.z, MvecTransLength, isoc.color);
                    // updateGMWrap(Gtot.x, Gtot.y, Gtot.z, isoc.color);
                    showTotalCurve = false;
                }

            } //loop over isochromates

            if (showTotalCurve) {
                Mtot.multiplyScalar(state.curveScale / nIsoc);
                updateFidWrap(Mtot.x, Mtot.z, Mtot.projectOnPlane(unitZvec).length(), white);

                // Gtot.multiplyScalar(1 / MaxGMvec * 1);
                Gtot.multiplyScalar(state.curveScale / nIsoc);
                Atot.multiplyScalar(state.curveScale / nIsoc);
                if(Atot.x)
                {
                    updateGMWrap(0.5, Atot.y, Atot.z,  "ADC");
                    // fidbox.style["backgroundColor"] = "rgb(255, 0, 100, 0.5)"; // remove the adc in the Signal figure
                }
                else
                {
                    updateGMWrap(0, Atot.y, Atot.z,  "ADC");
                    // fidbox.style["backgroundColor"] = "transparent"; // remove the adc in the Signal figure
                }
                updateGMWrap(Gtot.x, Gtot.y, Gtot.z,  "RFandGM");
            }

            doStats && stats.update();

            if (!frameFixed) {
                if (state.FrameB0) { framePhase += state.B0 * state.Gamma * dt }
                else if (state.FrameB1) { framePhase += state.B1freq * dt };
                scene.rotation.z = framePhase; // continuous, even if B0 or B1freq change.
            }


        } //if (!pause)
        renderer.render(scene, camera);

    } //animate

    init();
    lastTime = window.performance.now(); //start timer
    then = lastTime;
    requestAnimationFrame(animate);
    // hideLoader();

} //launchApp

//main
window.addEventListener('resize', onResize, false);
window.onload = launchApp;
