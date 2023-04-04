package simplernn

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

type RNN struct {
	input   int
	hidden  int
	output  int
	wih     *mat64.Dense // input to hidden weights
	whh     *mat64.Dense // hidden to hidden weights
	who     *mat64.Dense // hidden to output weights
	bh      *mat64.Dense // hidden bias
	bo      *mat64.Dense // output bias
	hprev   *mat64.Dense
	hiddens *mat64.Dense

	n         int
	TotalLoss float64
}

func NewRNN(input, hidden, output int) *RNN {

	result := &RNN{}
	result.input = input
	result.hidden = hidden
	result.output = output

	result.wih = randomMatrix(hidden, input)
	result.wih.Scale(0.01, result.wih)

	result.whh = randomMatrix(hidden, hidden)
	result.whh.Scale(0.01, result.whh)

	result.who = randomMatrix(output, hidden)
	result.who.Scale(0.01, result.who)

	result.bh = mat64.NewDense(hidden, 1, nil) // hidden bias
	result.bo = mat64.NewDense(output, 1, nil) // output bias

	result.hiddens = mat64.NewDense(hidden, 1, nil)
	return result
}

func (r *RNN) Forward(input, hprev []float64) *mat64.Dense {

	inputs := mat64.NewDense(r.input, 1, input)
	hiddenInputs := dot(r.wih, inputs)

	hp := mat64.NewDense(r.hidden, 1, hprev)
	hidenhp := dot(r.whh, hp)

	hidden := &mat64.Dense{}
	hidden.Add(hiddenInputs, hidenhp)
	hidden.Add(hidden, r.bh)

	r.hiddens = apply(hidden, math.Tanh)

	finalInputs := dot(r.who, r.hiddens)
	finalInputs = add(finalInputs, r.bo)

	finalOutputs := apply(finalInputs, math.Tanh)
	return finalOutputs
}

func (r *RNN) Train(input, hprev, target []float64, learningRate float64) {

	inputs := mat64.NewDense(r.input, 1, input)
	targets := mat64.NewDense(len(target), 1, target)
	hp := mat64.NewDense(r.hidden, 1, hprev)

	finalOutputs := r.Forward(input, hprev)

	for i := 0; i < r.output; i++ {
		r.TotalLoss += 0.5 * math.Pow(target[i]-finalOutputs.At(i, 0), 2)
	}

	outputErrors := subtract(finalOutputs, targets)

	hiddenErrors := dot1(r.who.T(), outputErrors)

	r.who = add2(r.who, scale2(learningRate, dot1(multiply2(outputErrors, tanhPrime(finalOutputs)), r.hiddens.T()))).(*mat64.Dense)
	r.wih = add2(r.wih, scale2(learningRate, dot1(multiply2(hiddenErrors, tanhPrime(r.hiddens)), inputs.T()))).(*mat64.Dense)
	r.whh = add2(r.whh, scale2(learningRate, dot1(multiply2(hiddenErrors, tanhPrime(r.hiddens)), hp.T()))).(*mat64.Dense)

	r.bh = add2(r.bh, scale2(learningRate, hiddenErrors)).(*mat64.Dense)
	r.bo = add2(r.bo, scale2(learningRate, outputErrors)).(*mat64.Dense)

	r.n++
}

func (r *RNN) Run(inputs, outputs [][]float64, learningRate float64, epoch, batch int) {
	for j := 0; j < epoch; j++ {
		hprev := make([]float64, r.hidden)
		for i := 0; i < len(inputs); i += batch {
			end := i + batch
			if end > len(inputs) {
				end = len(inputs)
			}
			batchX := inputs[i:end]
			batchY := outputs[i:end]
			for o, input := range batchX {
				r.Train(input, hprev, batchY[o], learningRate)
				copy(hprev, input)
				if (r.n+1)%(1000) == 0 {
					avgLoss := r.TotalLoss / float64(r.n)
					fmt.Printf("Epoch %d: Loss = %.4f \n", r.n+1, avgLoss)
				}
			}
		}
	}
}
