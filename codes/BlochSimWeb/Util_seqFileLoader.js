"use strict";

// This file contains classes to parse all pulseq sections except [EXTENSIONS]

class Version {
    constructor(lines) {
        if (lines.length != 3) {
            throw "[VERSION] must contain 3 lines for major, minor and revision";
        }
        let major = null;
        let minor = null;
        let revision = null;

        for (let line of lines) {
            let [name, value] = line.split(" ", 2);
            if (name == "major") {
                major = parseInt(value, 10);
            } else if (name == "minor") {
                minor = parseInt(value, 10);
            } else if (name == "revision") {
                revision = parseInt(value, 10);
            }
        }
        // Storing the version as int for comparisons like: if (version < 140)
        this.version = major * 100 + minor * 10 + revision;
    }

    display() {
        let div = document.createElement("div");
        div.innerText = this.version;
        div.classList.add("version");
        div.setAttribute("title", "Pulseq file format version");
        return div;
    }
}

class Definitions {
    constructor(lines) {
        this.definitions = {};

        for (let line of lines) {
            let i = line.indexOf(" ");
            let key = [line.slice(0, i)];
            if (key in this.definitions) {
                throw `[DEFINITIONS] contains key "${key}" more than once`;
            }
            this.definitions[key] = line.slice(i + 1);
        }

        // TODO: parse raster time definitions needed for simulation
    }

    display() {
        let table = document.createElement("table");

        let thead = document.createElement("thead");
        table.appendChild(thead);

        function addTh(text, tooltip) {
            let th = document.createElement("th");
            thead.appendChild(th);
            th.innerText = text;
            th.setAttribute("title", tooltip);
        }

        addTh("Key", "Name of the definition");
        addTh("Value", "Value of the definition");

        for (let key in this.definitions) {
            let values = this.definitions[key];

            let tr = table.insertRow();
            tr.insertCell().innerText = key;
            tr.insertCell().innerText = values;
        }
        return table;
    }
}

class Shapes {
    constructor(lines) {
        this.shapes = {};
        let shape = null;

        for (let line of lines) {
            if (line.toLowerCase().startsWith("shape_id")) {
                let id = parseInt(line.slice(8), 10);
                if (id in this.shapes) {
                    throw `Shape with id ${id} is defined more than once`;
                }
                shape = { count: -1, samples: [] };
                this.shapes[id] = shape;
            } else if (shape.count == -1) {
                // Spec says its called "num_uncompressed", but test files contain
                // "num_samples" so we don't care but assume this is what we parse
                shape.count = parseInt(line.split(" ", 2)[1]);
            } else {
                // Could check if a) shape was created, b) count is already set
                shape.samples.push(parseFloat(line));
            }
        }
        for (var key in this.shapes)
        {
            let shape = this.shapes[key];
            let temp = [];
            // compression back to derivative
            for (let idx = 0; idx < shape.samples.length; idx++)
            {
                let value = shape.samples[idx];
                if (temp.length >= 2 )
                {
                    if (shape.samples[idx-1] == shape.samples[idx-2])
                    {
                        for (let i = 0; i < value; i++)
                        {
                            temp.push(shape.samples[idx-1])
                        }
                    }
                    else
                    {
                        temp.push(value);
                    }
                }
                else
                {
                    temp.push(value);
                }
            }
            // derivative back to shape
            let res = [];
            let temp_value = 0;
            for (let i=0; i<temp.length; i+=1)
            {
                temp_value +=  temp[i];
                res.push(temp_value)
            }
            shape.samples = res;
        }
    }

    display() {
        let table = document.createElement("table");

        let thead = document.createElement("thead");
        table.appendChild(thead);

        function addTh(text, tooltip) {
            let th = document.createElement("th");
            thead.appendChild(th);
            th.innerText = text;
            th.setAttribute("title", tooltip);
        }

        addTh("ID", "ID of the event");
        addTh("sample_count", "Number of samples in the uncompressed shape");
        addTh("samples", "Sample values, possibly compressed");

        for (let shape_id in this.shapes) {
            let shape = this.shapes[shape_id];

            let tr = table.insertRow();
            tr.insertCell().innerText = shape_id;
            tr.insertCell().innerText = shape.count;
            tr.insertCell().innerText = shape.samples.join("\n");
        }
        return table;
    }
}

class IdSection {
    constructor(lines, sec_def) {
        this.sec_def = sec_def;
        this.events = {};

        for (let line of lines) {
            const ids = line.split(" ").filter((x) => x.length > 0);
            let id = parseInt(ids[0], 10);
            if (id in this.events) {
                throw `Event with ID ${id} is defined more than once`;
            }
            if (ids.length != sec_def.length + 1) {
                throw `Expected ${sec_def.length + 1} IDs per event, found ${
                    ids.length
                }`;
            }

            let event = {};
            for (let i = 0; i < sec_def.length; i++) {
                event[sec_def[i][0]] = sec_def[i][1](ids[i + 1]);
            }
            this.events[id] = event;
        }
    }

    display() {
        let sec_def = this.sec_def;
        let table = document.createElement("table");

        let thead = document.createElement("thead");
        table.appendChild(thead);

        let th = document.createElement("th");
        th.innerText = "ID";
        th.setAttribute("title", "ID of the event");
        thead.appendChild(th);

        for (let i = 0; i < sec_def.length; i++) {
            let th = document.createElement("th");
            th.innerText = sec_def[i][0];
            th.setAttribute("title", sec_def[i][2]);
            thead.appendChild(th);
        }

        for (let elem_id in this.events) {
            let tr = table.insertRow();
            tr.insertCell().innerText = elem_id;

            let elem = this.events[elem_id];
            for (let i = 0; i < sec_def.length; i++) {
                tr.insertCell().innerText = elem[sec_def[i][0]];
            }
        }
        return table;
    }
}

class Blocks extends IdSection {
    constructor(lines) {
        super(lines, [
            ["delay", (x) => parseInt(x, 10), "ID of delay event"],
            ["rf", (x) => parseInt(x, 10), "ID of RF event"],
            ["gx", (x) => parseInt(x, 10), "ID of gradient event (x channel)"],
            ["gy", (x) => parseInt(x, 10), "ID of gradient event (y channel)"],
            ["gz", (x) => parseInt(x, 10), "ID of gradient event (z channel)"],
            ["adc", (x) => parseInt(x, 10), "ID of ADC event"],
            // ["ext", (x) => parseInt(x, 10), "ID of extension table entry"],
        ]);
    }
}

class Rf extends IdSection {
    constructor(lines) {
        super(lines, [
            ["amp", parseFloat, "Amplitude [Hz]"],
            ["mag_id", (x) => parseInt(x, 10), "ID of magnitude shape"],
            ["phase_id", (x) => parseInt(x, 10), "ID of phase shape"],
            ["delay", (x) => parseInt(x, 10), "Delay before pulse [µs]"],
            ["freq", parseFloat, "Frequency offset [Hz]"],
            ["phase", parseFloat, "Phase offset [rad]"],
        ]);
    }
}

class Gradients extends IdSection {
    constructor(lines) {
        super(lines, [
            ["amp", parseFloat, "Amplitude [Hz/m]"],
            ["shape_id", (x) => parseInt(x, 10), "ID of gradient shape"],
            ["delay", (x) => parseInt(x, 10), "Delay before gradient [µs]"],
        ]);
    }
}

class Trap extends IdSection {
    constructor(lines) {
        super(lines, [
            ["amp", parseFloat, "Amplitude [Hz/m]"],
            ["rise", (x) => parseFloat(x), "Rise time [µs]"],
            ["flat", (x) => parseFloat(x), "Flat time [µs]"],
            ["fall", (x) => parseFloat(x), "Fall time [µs]"],
            ["delay", (x) => parseInt(x, 10), "Delay before gradient [µs]"],
        ]);
    }
}

class Adc extends IdSection {
    constructor(lines) {
        super(lines, [
            ["num", (x) => parseInt(x, 10), "Number of samples"],
            ["dwell", (x) => parseFloat(x), "ADC dwell time [ns]"],
            ["delay", (x) => parseInt(x, 10), "Delay before first sample [µs]"],
            ["freq", parseFloat, "Frequency offset [Hz]"],
            ["phase", parseFloat, "Phase offset [rad]"],
        ]);
    }
}

class Delays extends IdSection {
    constructor(lines) {
        super(lines, [
            ["delay", (x) => parseInt(x, 10), "Duration of delay [µs]"],
        ]);
    }
}

// function scientificNotationConversion(x)
// {
//     var num = 0;
//     var times = 0;
//     if (x.search("e") != -1)
//     {
//         num =  parseFloat(x, 10)
//         console.log(num)
//         x = x.replace(num.toString(),"")
//         x = x.replace("e","")
//         times =  parseInt(x, 10)
//         return num * Math.pow(10, times)
//     }
//     else
//     {
//         return  parseFloat(x, 10);
//     }


// }
