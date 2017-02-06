/*
var fs = require('fs');
var convnetjs = require('convnetjs');
var cnnutil = require('convnetjs/build/util');
global.convnetjs = convnetjs;
global.cnnutil = cnnutil;
var deepqlearn = require('convnetjs/build/deepqlearn');
*/

class Brain {
    constructor(width, height) {
        const num_inputs = width * height;
        const num_actions = num_inputs; // can click on each field
        const temporal_window = 0; // no history needed (?)
        const network_size = num_inputs*temporal_window + num_actions*temporal_window + num_inputs;

        let layer_defs = [];
        layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:num_inputs});
        //layer_defs.push({type:'conv', sx:3, filters:8, stride:1, activation:'relu'});
        layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
        layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
        layer_defs.push({type:'regression', num_neurons:num_actions});

        const tdtrainer_options = {learning_rate:0.001, momentum:0.0, batch_size:64, l2_decay:0.01};

        let opt = {};
        opt.temporal_window = 0;
        opt.experience_size = 30000;
        opt.start_learn_threshold = 1000;
        opt.gamma = 0.7;
        opt.learning_steps_total = 200000;
        opt.learning_steps_burnin = 3000;
        opt.epsilon_min = 0.05;
        opt.epsilon_test_time = 0.05;
        opt.layer_defs = layer_defs;
        opt.tdtrainer_options = tdtrainer_options;

        this.brain = new deepqlearn.Brain(num_inputs, num_actions, opt);
    }

    load() {
        if (fs.exists('brain.json')) {
            this.brain.value_net.fromJSON(JSON.parse(fs.readFileSync('brain.json')));
        }
    }

    save() {
        fs.writeFileSync('brain.json', JSON.stringify(this.brain.value_net.toJSON()));
    }

    play(cf) {
        let input = cf.field.arr.map(x => x === null ? -1 : x);
        let action = this.brain.forward(input);
        let [x, y] = cf.field.reverseIndex(action);
        let result = cf.click(action);
        if (result === false) {
            // boom
            this.brain.backward(-1);
        } else if (result === 0) {
            // punish no-ops
            this.brain.backward(0);
        } else {
            this.brain.backward(1);
        }
        return action;
    }

    playLoop(cf) {
        while (cf.result === null) {
            this.play(cf);
        }
        return cf.result;
    }
}

class Field {
    constructor(width, height) {
        this.arr = [];
        this.width = width;
        this.height = height;
        for (var i = 0; i < width * height; i++) {
            this.arr.push(null);
        }
    }

    index(x, y) {
        return x + y * this.width;
    }

    value(x, y) {
        return valueIndex(this.index(x, y));
    }

    valueIndex(index) {
        return this.arr[index];
    }

    update(x, y, v) {
        return this.updateIndex(this.index(x, y), v);
    }

    updateIndex(index, v) {
        const prev = this.arr[index];
        this.arr[index] = v;
        return prev;
    }

    randomIndex() {
      return Math.floor(Math.random() * this.arr.length);
    }

    reverseIndex(index) {
        let x = index % this.width;
        let y = (index - x) / this.width;
        return [x, y];
    }

    neighbors(index) {
        let [x, y] = this.reverseIndex(index);
        let neighbors = [];
        for (var i = Math.max(0, x - 1); i < Math.min(this.width, x + 2); i++) {
            for (var j = Math.max(0, y - 1); j < Math.min(this.height, y + 2); j++) {
                let idx = this.index(i, j);
                if (idx !== index) {
                    neighbors.push(idx);
                }
            }
        }
        return neighbors;
    }

    print(f) {
        for (var r = 0; r < this.height; r++) {
            let row = this.arr.slice(r * this.width, (r + 1) * this.width);
            let line = row.map((value, col) => f(value, r * this.width + col)).join('');
            console.log(line);
        }
    }
}

class Minefield {
    constructor(width, height, mines) {
        this.field = new Field(width, height);
        while (mines > 0) {
            const prev = this.field.updateIndex(this.field.randomIndex(), true);
            if (!prev) { mines--; }
        }
        //this.field.updateIndex(5, true);
        //this.field.updateIndex(8, true);
        //this.field.updateIndex(13, true);
        //this.field.updateIndex(20, true);
        //this.field.updateIndex(22, true);
        //this.field.updateIndex(50, true);
        //this.field.updateIndex(51, true);
        //this.field.updateIndex(59, true);
        //this.field.updateIndex(70, true);
    }

    get width() { return this.field.width; }
    get height() { return this.field.height; }

    count(index) {
        return [index].concat(this.field.neighbors(index)).map(i => this.field.valueIndex(i) ? 1 : 0).reduce((s, x) => s + x, 0);
    }

    print() {
        this.field.print(x => x ? 'x' : '.');
    }
}

class Countfield {
    constructor(minefield) {
        this.minefield = minefield;
        this.field = new Field(minefield.width, minefield.height);
        this.gameover = false;
    }

    click(index) {
        if (this.field.valueIndex(index) !== null) {
            // already known field
            return 0;
        }

        if (this.minefield.field.valueIndex(index)) {
            // mine
            this.gameover = true;
            return false;
        }

        // new field, without mine, reveal field
        let queue = [index];
        let revealed = 0;
        while (queue.length > 0) {
            let i = queue.pop();
            if (this.field.valueIndex(i) !== null) {
                continue;
            }

            let count = this.minefield.count(i);
            if (count === 0) {
                // need to potentially reveal all other 8 fields around us
                queue = queue.concat(this.field.neighbors(i));
            }
            this.field.updateIndex(i, count);
            revealed++;
        }

        return revealed;
    }

    print() {
        this.field.print((x, i) => {
            if (this.minefield.field.valueIndex(i)) {
                return 'x';
            }
            return x !== null ? x : '.'
        });
    }

    get done() {
        const hidden = this.field.arr.filter(x => x === null).length;
        const mines = this.minefield.field.arr.filter(x => x).length;
        return hidden <= mines;
    }

    get result() {
        if (this.gameover) {
            return false;
        }
        if (this.done) {
            return true;
        }
        return null;
    }
}

class Screen {
    constructor() {
        this.canvas = document.getElementById('minefield');
        this.ctx = this.canvas.getContext('2d');
    }

    draw(cf) {
        this.ctx.font = '48px sans-serif';
        this.ctx.textBaseline = 'hanging';
        for (var i = 0; i < cf.field.arr.length; i++) {
            let [x, y] = cf.field.reverseIndex(i);
            this.ctx.save();
            this.ctx.translate(x * 50, y * 50);


            let v = cf.field.valueIndex(i);
            if (v !== null) {
                this.ctx.fillStyle = '#ddd';
                this.ctx.fillRect(0, 0, 50, 50);

                if (v !== 0) {
                    this.ctx.fillStyle = '#000';
                    this.ctx.fillText(v, 10, 10, 50);
                }
            } else {
                this.ctx.fillStyle = '#bbb';
                this.ctx.fillRect(0, 0, 50, 50);
            }

            this.ctx.strokeStyle = '#000';
            this.ctx.strokeRect(0, 0, 50, 50);

            this.ctx.restore();
        }

        for (var i = 0; i < cf.minefield.field.arr.length; i++) {
            let [x, y] = cf.minefield.field.reverseIndex(i);
            this.ctx.save();
            this.ctx.translate(x * 50, y * 50);


            let v = cf.minefield.field.valueIndex(i);
            if (v) {
                this.ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
                this.ctx.fillRect(0, 0, 50, 50);
            }
            this.ctx.restore();
        }
    }

    drawAction(cf, action) {
        let [x, y] = cf.field.reverseIndex(action);
        this.ctx.save();
        this.ctx.translate(x * 50, y * 50);
        this.ctx.fillStyle = 'rgba(0, 0, 255, 0.3)';
        this.ctx.fillRect(0, 0, 50, 50);
        this.ctx.restore();
    }
}

var reward_graph = new cnnvis.Graph();
let draw_stats = () => {
    var canvas = document.getElementById("vis_canvas");
    var ctx = canvas.getContext("2d");
    var W = canvas.width;
    var H = canvas.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var b = br.brain;
    var netin = b.last_input_array;
    ctx.strokeStyle = "rgb(0,0,0)";
    //ctx.font="12px Verdana";
    //ctx.fillText("Current state:",10,10);
    ctx.lineWidth = 10;
    ctx.beginPath();
    for(var k=0,n=netin.length;k<n;k++) {
        ctx.moveTo(10+k*12, 120);
        ctx.lineTo(10+k*12, 120 - netin[k] * 100);
    }
    ctx.stroke();

    //if(clock % 2 === 0) {
        reward_graph.add(clock/200, b.average_reward_window.get_average());
        var gcanvas = document.getElementById("graph_canvas");
        reward_graph.drawSelf(gcanvas);
    //}
}

let draw_net = () => {
    var canvas = document.getElementById("net_canvas");
    var ctx = canvas.getContext("2d");
    var W = canvas.width;
    var H = canvas.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var L = window.br.brain.value_net.layers;
    var dx = (W - 50)/L.length;
    var x = 10;
    var y = 40;
    ctx.font="12px Verdana";
    ctx.fillStyle = "rgb(0,0,0)";
    ctx.fillText("Value Function Approximating Neural Network:", 10, 14);
    for(var k=0;k<L.length;k++) {
        if(typeof(L[k].out_act)==='undefined') continue; // maybe not yet ready
        var kw = L[k].out_act.w;
        var n = kw.length;
        var dy = (H-50)/n;
        ctx.fillStyle = "rgb(0,0,0)";
        ctx.fillText(L[k].layer_type + "(" + n + ")", x, 35);
        for(var q=0;q<n;q++) {
            var v = Math.floor(kw[q]*100);
            if(v >= 0) ctx.fillStyle = "rgb(0,0," + v + ")";
            if(v < 0) ctx.fillStyle = "rgb(" + (-v) + ",0,0)";
            ctx.fillRect(x,y,10,10);
            y += 12;
            if(y>H-25) { y = 40; x += 12};
        }
        x += 50;
        y = 40;
    }
}


window.br = new Brain(9, 9);
window.clock = 0;
let sc = new Screen();

let mf = new Minefield(9, 9, 10);
let cf = new Countfield(mf);

setInterval(() => {
    clock++;
    if (cf.result === null) {
        let action = br.play(cf);
        sc.draw(cf);
        sc.drawAction(cf, action);
        draw_stats();
        draw_net();
    } else {
        console.log(cf.result ? 'won' : 'lost');
        mf = new Minefield(9, 9, 10);
        cf = new Countfield(mf);
    }
}, 10);

document.getElementById('step').onclick = () => {
    let action = br.play(cf);
    sc.draw(cf);
    sc.drawAction(cf, action);
};
