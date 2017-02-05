var fs = require('fs');
var convnetjs = require('convnetjs');
var cnnutil = require('convnetjs/build/util');
global.convnetjs = convnetjs;
global.cnnutil = cnnutil;
var deepqlearn = require('convnetjs/build/deepqlearn');
/*

var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
layer_defs.push({type:'fc', num_neurons:5, activation:'sigmoid'});
layer_defs.push({type:'regression', num_neurons:1});
var net = new convnetjs.Net();
net.makeLayers(layer_defs);
 
var x = new convnetjs.Vol([0.5, -1.3]);
 
// train on this datapoint, saying [0.5, -1.3] should map to value 0.7:
// note that in this case we are passing it a list, because in general
// we may want to  regress multiple outputs and in this special case we 
// used num_neurons:1 for the regression to only regress one.
var trainer = new convnetjs.SGDTrainer(net, 
              {learning_rate:0.01, momentum:0.0, batch_size:1, l2_decay:0.001});
trainer.train(x, [0.7]);
 
// evaluate on a datapoint. We will get a 1x1x1 Vol back, so we get the
// actual output by looking into its 'w' field:
var predicted_values = net.forward(x);
console.log('predicted value: ' + predicted_values.w[0]);
*/

function rand(min, max) {
  return Math.floor(Math.random() * (max - min) + min);
}

function count(minefield, x, y) {
    let mines = 0;
    for (var i = Math.max(0, x - 1); i < Math.min(minefield.length, x + 2); i++) {
        for (var j = Math.max(0, y - 1); j < Math.min(minefield[i].length, y + 2); j++) {
            if (minefield[i][j]) {
                mines++;
            }
        }
    }
    return mines;
}

function createMinefield(size_x, size_y) {
    let mine_count = rand(2, 4);
    let minefield = [];

    for (var i = 0; i < size_x; i++) {
        minefield[i] = [];
        for (var j = 0; j < size_y; j++) {
            minefield[i][j] = false;
        }
    }

    while (mine_count > 0) {
        let mine_x = rand(0, size_x);
        let mine_y = rand(0, size_y);
        if (!minefield[mine_x][mine_y]) {
            minefield[mine_x][mine_y] = true;
            mine_count--;
        }
    }

    return minefield;
}

function createCountField(minefield) {
    let countfield = [];
    for (var i = 0; i < minefield.length; i++) {
        countfield[i] = [];
        let col = minefield[i];
        for (var j = 0; j < col.length; j++) {
            let c = count(minefield, i, j);
            countfield[i][j] = c;
        }
    }
    return countfield;
}

function drawMinefield(minefield) {
    for (var i = 0; i < minefield.length; i++) {
        var col = minefield[i];
        console.log(col.map(row => row ? 'x' : '.').join(''));
    }
}

function drawCountfield(countfield) {
    for (var i = 0; i < countfield.length; i++) {
        var col = countfield[i];
        console.log(col.map(r => r !== null ? r : '.').join(''));
    }
}

function neighbors(minefield, x, y, f) {
    for (var i = Math.max(0, x - 1); i < Math.min(minefield.length, x + 2); i++) {
        for (var j = Math.max(0, y - 1); j < Math.min(minefield[i].length, y + 2); j++) {
            f(i, j);
        }
    }
}

function field(minefield, f) {
    for (var i = 0; i < minefield.length; i++) {
        for (var j = 0; j < minefield[i].length; j++) {
            f(i, j, minefield[i][j]);
        }
    }
}

function initCountfield(minefield) {
    return minefield.map(col => col.map(() => null));
}

function simClick(minefield, countfield, x, y) {
    if (minefield[x][y]) {
        throw "boom";
    }

    let queue = [[x, y]];

    while (queue.length > 0) {
        let [i, j] = queue.pop();

        if (countfield[i][j] !== null) {
            //console.log('already know ' + i + ', ' + j);
            continue;
        }

        let c = count(minefield, i, j);
        //console.log('count of ' + i + ', ' + j + ': ' + c);

        if (c === 0) {
            // need to potentially reveal all other 8 fields around us
            neighbors(minefield, i, j, (a, b) => queue.push([a, b]));
        }

        countfield[i][j] = c;
    }

    return countfield;
}

function isDone(minefield, countfield) {
    // we're done when there is less or equal unclicked fields than there is
    // mines
    let mines = 0;
    field(minefield, (i, j, m) => { if (m) { mines++; } });

    let unclicked = 0;
    field(countfield, (i, j, m) => { if (m === null) { unclicked++; } });

    return unclicked <= mines;
}

function flatten(field) {
    return field.reduce((s, col) => s.concat(col), []);
}

var num_inputs = 25; // 9 eyes, each sees 3 numbers (wall, green, red thing proximity)
var num_actions = 25; // 5 possible angles agent can turn
var temporal_window = 0; // amount of temporal memory. 0 = agent lives in-the-moment :)
var network_size = num_inputs*temporal_window + num_actions*temporal_window + num_inputs;

var layer_defs = [];
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:network_size});
layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
layer_defs.push({type:'fc', num_neurons: 50, activation:'relu'});
layer_defs.push({type:'regression', num_neurons:num_actions});

var tdtrainer_options = {learning_rate:0.001, momentum:0.0, batch_size:64, l2_decay:0.01};

var opt = {};
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

let brain = new deepqlearn.Brain(25, 25);

if (fs.exists('brain.json')) {
    brain.value_net.fromJSON(JSON.parse(fs.readFileSync('brain.json')));
}

let won = 0;
let lost = 0;

for (var i = 0; i < 100; i++) {
    if (i % 100 === 0) {
        console.log('iter: ' + i + ', won: ' + won + ', lost: ' + lost);
    }
    try {
        let mf = createMinefield(5, 5);
        let cf = initCountfield(mf);
        //drawMinefield(mf);
        //drawCountfield(cf);

        while (true) {
            let action = brain.forward(flatten(cf));

            //console.log(action);
            let row = Math.floor(action / mf.length);
            let col = action - row * mf.length;
            //console.log(row, col);
            if (cf[row][col] !== null) {
                brain.backward(-1);
            } else {
                cf = simClick(mf, cf, row, col);
                //drawCountfield(cf);
                brain.backward(1);
                if (isDone(mf, cf)) {
                    //console.log('won :-)');
                    brain.backward(10);
                    won++;
                    break;
                }
            }
        }
    } catch (e) {
        //console.log('lost :-(');
        lost++;
        brain.backward(-10);
    }
}
console.log('won: ' + won);
console.log('lost: ' + lost);

fs.writeFileSync('brain.json', JSON.stringify(brain.value_net.toJSON()));

let mf = createMinefield(5, 5);
let cf = initCountfield(mf);
drawMinefield(mf);

try {
    while (true) {
        console.log('');
        drawCountfield(cf);
        let action = brain.forward(flatten(cf));

        let row = Math.floor(action / mf.length);
        let col = action - row * mf.length;
        console.log('click on ' + row + ', ' + col);

        cf = simClick(mf, cf, row, col);
        if (isDone(mf, cf)) {
            console.log('done');
            break;
        }

    }
} catch (e) {
    drawMinefield(cf);
    console.log('boom');
}
