"use strict";

var _gpu = _interopRequireDefault(require("gpu.js"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

//test for gpu.js
var matrixSize = 512;
var allowChangeOutput = true;
var a = new Array(matrixSize * matrixSize);
var b = new Array(matrixSize * matrixSize);
a = splitArray(fillArrayRandom(a), matrixSize);
b = splitArray(fillArrayRandom(b), matrixSize);

function fillArrayRandom(array) {
  for (var i = 0; i < array.length; i++) {
    array[i] = Math.random();
  }

  return array;
}

function splitArray(array, part) {
  var result = [];

  for (var i = 0; i < array.length; i += part) {
    result.push(array.slice(i, i + part));
  }

  return result;
}

var gpu = new _gpu.default();
var multiplyMatrix = gpu.createKernel(function (a, b) {
  var sum = 0;

  for (var i = 0; i < 512; i++) {
    sum += a[this.thread.y][i] * b[i][this.thread.x];
  }

  return sum;
}).setOutput([512, 512]);
var c = multiplyMatrix(a, b);
console.log(c);
