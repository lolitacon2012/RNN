"use strict";

var N = 4; // length of s

var M = 7; // total numbers of fingers

var s = "CGPC"; // input sequence

var dp = []; // 2-D array, stores results.

var min = Number.MIN_SAFE_INTEGER;

var CGP = function CGP(sequence, max_finger) {
  if (max_finger < 0) {
    return min;
  }

  console.log("SEQUENCE = " + sequence + ", Finger = " + max_finger); //c:2, g:0, p:5;

  var sequence_remaining = sequence.length;

  if (sequence_remaining == 1) {
    var result = 0;
    console.log("Current opponent = " + sequence);

    switch (sequence) {
      case "C":
        if (max_finger == 0) {
          result = 1;
        } else if (max_finger == 2 || max_finger == 5) {
          result = 0;
        } else {
          result = min;
        }

        break;

      case "G":
        if (max_finger == 0 || max_finger == 2) {
          result = 0;
        } else if (max_finger == 5) {
          result = 1;
        } else {
          result = min;
        }

        break;

      case "P":
        if (max_finger == 5 || max_finger == 0) {
          result = 0;
        } else if (max_finger == 2) {
          result = 1;
        } else {
          result = min;
        }

        break;
    }

    if (!dp[sequence.length]) {
      dp[sequence.length] = [];
    }

    dp[sequence.length][max_finger] = result;
    console.log("For " + sequence + " result = " + result);
    return result;
  } else {
    if (!!dp[sequence.length] && dp[sequence.length][max_finger] != undefined) {
      return dp[sequence.length][max_finger];
    }

    var sequence_previous = sequence.slice(0, sequence_remaining - 1);
    var current_opponent = sequence.charAt(sequence_remaining - 1);
    var _result = 0;
    console.log("Current opponent = " + current_opponent);

    switch (current_opponent) {
      case "C":
        _result = Math.max(CGP(sequence_previous, max_finger) + 1, CGP(sequence_previous, max_finger - 2), CGP(sequence_previous, max_finger - 5));
        break;

      case "G":
        _result = Math.max(CGP(sequence_previous, max_finger - 5) + 1, CGP(sequence_previous, max_finger), CGP(sequence_previous, max_finger - 2));
        break;

      case "P":
        _result = Math.max(CGP(sequence_previous, max_finger - 2) + 1, CGP(sequence_previous, max_finger - 5), CGP(sequence_previous, max_finger));
        break;
    } //save result


    if (!dp[sequence.length]) {
      dp[sequence.length] = [];
    }

    dp[sequence.length][max_finger] = _result;
    console.log("For " + sequence + " result = " + _result);
    return _result;
  }
};

console.log(CGP(s, M));
