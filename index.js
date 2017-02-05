var fs = require('fs');
var convnetjs = require('convnetjs');
var cnnutil = require('convnetjs/build/util');
global.convnetjs = convnetjs;
global.cnnutil = cnnutil;
var deepqlearn = require('convnetjs/build/deepqlearn');

class Brain {
    constructor(width, height) {
        const num_inputs = width * height;
        const num_actions = num_inputs; // can click on each field
        const temporal_window = 0; // no history needed (?)
        const network_size = num_inputs*temporal_window + num_actions*temporal_window + num_inputs;

        let layer_defs = [];
        layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:network_size});
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

        this.brain = new deepqlearn.Brain(num_inputs, num_actions);
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
        let action = this.brain.forward(cf.field);
        let result = cf.click(action);
        if (result === false) {
            // boom
            this.brain.backward(-10);
        } else {
            this.brain.backward(result);
        }
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

    neighbors(index) {
        let x = index % this.width;
        let y = (index - x) / this.width;
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

let br = new Brain(9, 9);
br.load();

for (var i = 0; i < 100; i++) {
    let mf = new Minefield(9, 9, 10);
    let cf = new Countfield(mf);

    console.log(br.playLoop(cf));
}

br.save();

let mf = new Minefield(9, 9, 10);
let cf = new Countfield(mf);
br.playLoop(cf);
cf.print();
console.log(cf.result);
