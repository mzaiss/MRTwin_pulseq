"use strict";

let fileHandle;
async function loadSeq(){
    [fileHandle] = await window.showOpenFilePicker({
        types: [{
            description: 'Sequence File',
            accept: {
                'Sequence/*': ['.seq']
            }
        }],
        multiple: false
    });
    const fileData = await fileHandle.getFile();
    return fileData.text()

}


function resize(data, speed)
{
    var temp = [];
    let t = 0;
    for (let i =0; i <data.length; i++)
    {
        t += data[i];
        if (i % speed == 0)
        {
            temp.push(t);
            t = 0;
        }
    }
    return temp;
}
let loaded_sequence = null;

class Pulseq {
    constructor(lines) {
        // Split input into sections
        let sections = {};
        let current_section = null;

        for (let line of lines) {
            if (line[0] == "[") {
                let name = line.slice(1, -1); // Remove "[" and "]"
                if (name in sections) {
                    throw `Section [${name}] is defined more than once`;
                }
                current_section = [];
                sections[name] = current_section;
            } else {
                current_section.push(line);
            }
        }

        // Parse sections
        for (let sec in sections) {
            let lines = sections[sec];
            switch (sec.toUpperCase()) {
                case "VERSION":
                    this.version = new Version(lines);
                    break;
                case "DEFINITIONS":
                    this.definitions = new Definitions(lines);
                    break;
                case "SHAPES":
                    this.shapes = new Shapes(lines);
                    break;
                case "BLOCKS":
                    this.blocks = new Blocks(lines);
                    break;
                case "RF":
                    this.rf = new Rf(lines);
                    break;
                case "GRADIENTS":
                    this.gradients = new Gradients(lines);
                    break;
                case "TRAP":
                    this.trap = new Trap(lines);
                    break;
                case "ADC":
                    this.adc = new Adc(lines);
                    break;
                case "DELAYS":
                    this.delays = new Delays(lines);
                    break;
                default:
                    console.log(`Ignored section [${sec}]`);
                    break;
            }
        }

    }


    getSeq(speed){
        var sequenceDist = {};
        var currentIDX = 0;
        sequenceDist[currentIDX] = {"delay":0,
                                    "RF":0,
                                    "ADC":0,
                                    "trap":
                                    {"Gx":0,"Gy":0,"Gz":0}};
        for (var eventID in this.blocks.events)
        {
            var res = this.blocks.events[eventID];
            if(res["delay"] != 0)
            {
                var delayValue = this.delays.events[res["delay"]]["delay"];
                sequenceDist[currentIDX]["delay"] = delayValue  /1000
            }
            if(res["rf"]!=0)
            {
                var rfPuls = this.rf.events[res["rf"]];
                var rfDist = {};
                rfDist["delay"] = rfPuls["delay"] /1000;
                var amp = rfPuls["amp"];
                var mag = this.shapes.shapes[rfPuls["mag_id"]]["samples"];
                var phase = this.shapes.shapes[rfPuls["phase_id"]]["samples"];
                var angle = [];
                var angleSum = 0;
                mag.forEach((val, i, arr) => {
                    let real = amp * mag[i] * Math.cos(2*Math.PI*phase[i]);
                    let comp = amp * mag[i] * Math.sin(2*Math.PI*phase[i]);
                    let temp = Math.pow(Math.pow(real,2) + Math.pow(comp,2),0.5)
                    angle.push(temp * 1e-6 *0.909*  360)
                    angleSum += angle[i]
                })
                console.log(angleSum)
                rfDist["angle"] = resize(angle,speed)
                rfDist["phase"] = rfPuls["phase"]/Math.PI;
                sequenceDist[currentIDX]["RF"] = rfDist
            }
            if(res["gx"]!=0)
            {
                var g = this.trap.events[res["gx"]];
                var gDist = {};
                gDist["delay"] = g["delay"] /1000;
                gDist["amplitude"] =resize(g["amp"]);
                gDist["period"] = (g["rise"] + g["flat"] + g["fall"])/1000000;
                sequenceDist[currentIDX]["trap"]["Gx"] = gDist;
            }
            if(res["gy"]!=0)
            {
                var g = this.trap.events[res["gy"]];
                var gDist = {};
                gDist["delay"] = g["delay"] /1000;
                gDist["amplitude"] = resize(g["amp"]);
                gDist["period"] = (g["rise"] + g["flat"] + g["fall"])/1000000;
                sequenceDist[currentIDX]["trap"]["Gy"] = gDist;
            }
            if(res["gz"]!=0)
            {
                var g = this.trap.events[res["gz"]];

                var gDist = {};
                gDist["delay"] = g["delay"] /1000;
                gDist["amplitude"] = resize(g["amp"]);
                gDist["period"] = (g["rise"] + g["flat"] + g["fall"])/1000000 ;
                sequenceDist[currentIDX]["trap"]["Gz"] = gDist;
            }
            if(res["adc"]!=0)
            {
                var adc = this.adc.events[res["adc"]];
                var adcDist = {};
                adcDist["delay"] = adc["delay"] /1000;
                adcDist["dwell"] = adc["dwell"] /1000;
                adcDist["num"] = adc["num"]
                adcDist["freq"] = adc["freq"];
                adcDist["phase"] = adc["phase"];

                // currentIDX+=1
                sequenceDist[currentIDX]["ADC"] = adcDist
            }
            currentIDX+=1;
            sequenceDist[currentIDX] = {"delay":0,
            "RF":0,
            "ADC":0,
            "trap":
            {"Gx":0,"Gy":0,"Gz":0}};
        }
        return sequenceDist
    }
}

function readString(input_string) {
    // Split into lines and remove comments, whitespace, empty lines
    let lines = input_string
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 0 && line[0] != "#");

    loaded_sequence = new Pulseq(lines);
    // console.log(loaded_sequence)
    return loaded_sequence
}
