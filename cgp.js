const M = 7; // 手指头总数
const s = "CGPC" // 对手出的序列
const N = s.length; // N就是对手出的顺序的长度
const dp = []; // 一个二维数组，其中dp[x][y]表示“以s的前x位序列和最多y个手指，最多能赢多少次”。初始状态该数组全部是undefined，随着递归的进行其中会被逐渐填上一些计算结果，这样能避免重复计算。
const min = Number.MIN_SAFE_INTEGER; // 取一个最小可能的整数，当某个问题不存在答案的时候（譬如输入为“C”然后手指最多100根，不可能有解）返回该值。

const CGP = (sequence, max_finger) => { // 主函数
    if(max_finger<0){ // 如果max_finger小于0，解不存在，返回min
        return min;
    }
    const sequence_remaining = sequence.length;
    if(sequence_remaining==1){ // 如果输入序列长度为1，而且能恰好用尽max_finger个手指头，根据出的结果返回0（平手）或者1（胜利），否则返回min（解不存在）
        let result = 0;
        switch(sequence){
            case "C": // 如果对手出剪刀
                if(max_finger == 0 ){
                    result = 1;// 如果此时还剩0个可用手指，只能出石头，结果就是胜利1次
                }else if(max_finger == 2 || max_finger == 5){// 如果此时还剩2个或者5个可用手指，只能平手或者输，结果就是胜利0次
                    result = 0;
                }else{ // 无解
                    result = min;
                }
                break;
            case "G": // 下面同理
            if(max_finger == 0 || max_finger == 2){
                result = 0;
            }else if(max_finger == 5){
                result = 1;
            }else{
                result = min;
            }
            break;
            case "P":
            if(max_finger == 5 || max_finger == 0){
                result = 0;
            }else if(max_finger == 2){
                result = 1;
            }else{
                result = min;
            }
            break;
        }
        // 将计算结果放入dp数组，供之后查阅
        if(!dp[sequence.length]){
            dp[sequence.length] = [];
        }
        dp[sequence.length][max_finger] = result;
        return result;
    }else{ // 如果输入序列长度大于1且max_finger大于0:

        if(!!dp[sequence.length] && dp[sequence.length][max_finger]!=undefined){ // 如果db数组中已经有了当前的计算结果，无需再次计算，直接返回结果
            return dp[sequence.length][max_finger];
        }
        const sequence_previous = sequence.slice(0,sequence_remaining-1); // 上一步面对的序列（即去掉当前序列最后一个字母）
        const current_opponent = sequence.charAt(sequence_remaining-1); // 当前对手出的是什么（即当前序列最后一个字母）
        let result = 0;
        switch (current_opponent){
            case "C": // 如果当前对手出剪刀

                /* 当前对手出剪刀的话，我们有3种出法： 
                   1、出石头，胜利。总的胜利次数 = “只允许最多max_finger个手指，面对对手sequence_previous的序列，最多能赢次数” + 1
                        *max_finger是因为当前这步耗费了0个手指，最后+1是因为当前胜利了。
                   2、出剪刀，平局。总的胜利次数 = “只允许最多max_finger-2个手指，面对对手sequence_previous的序列，最多能赢次数” + 0
                        *max_finger-2是因为当前这步耗费了2个手指，最后+0是因为当前没有胜利。
                   3、出布，失败。总的胜利次数 = “只允许最多max_finger-5个手指，面对对手sequence_previous的序列，最多能赢次数” + 0
                        *max_finger-5是因为当前这步耗费了5个手指，最后+0是因为当前没有胜利。
                    
                    在这里你能发现，这个问题可以被一步步分解成小问题，第N步的结果取决于之前的结果。所以我们实际上在不停地重复利用CGP这个函数，并且
                    引入了db数组来储存之前计算过的结果以提高运行速度。
                */
                result = Math.max(CGP(sequence_previous,max_finger)+1,CGP(sequence_previous,max_finger-2),CGP(sequence_previous,max_finger-5));
                break;
            case "G": // 以下同理
                result = Math.max(CGP(sequence_previous,max_finger-5)+1,CGP(sequence_previous,max_finger),CGP(sequence_previous,max_finger-2));
                break;
            case "P":
                result = Math.max(CGP(sequence_previous,max_finger-2)+1,CGP(sequence_previous,max_finger-5),CGP(sequence_previous,max_finger));
                break;
        }
        // 将计算结果放入dp数组，供之后查阅
        if(!dp[sequence.length]){
            dp[sequence.length] = [];
        }
        dp[sequence.length][max_finger] = result;
        return result;
    }
}

//打印结果
console.log(CGP(s, M));