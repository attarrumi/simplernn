package simplernn

/*
#include <math.h>

double Exp(double x) {
	return exp(x);
}
*/
import "C"
import (
	"math"
	"math/rand"

	"github.com/attarrumi/goa"
	"github.com/gonum/matrix/mat64"
)

func MapInput(input string) (charToIndex map[rune]int, indexToChar map[int]rune) {
	charToIndex = make(map[rune]int)
	indexToChar = make(map[int]rune)

	// find unique chars in input and map them to/from ints
	uniques := 0
	for _, ch := range input {
		if _, ok := charToIndex[ch]; !ok {
			charToIndex[ch] = uniques
			indexToChar[uniques] = ch
			uniques++
		}
	}

	return charToIndex, indexToChar
}

func dot(a *mat64.Dense, b *mat64.Dense) *mat64.Dense {
	var c mat64.Dense
	c.Mul(a, b)
	return &c
}

func dot1(a, b mat64.Matrix) *mat64.Dense {
	result := &mat64.Dense{}

	result.Mul(a, b)

	return result
}
func add(a *mat64.Dense, b *mat64.Dense) *mat64.Dense {
	rows, cols := a.Dims()
	c := mat64.NewDense(rows, cols, nil)

	c.Add(a, b)
	return c
}

func apply(a *mat64.Dense, f func(float64) float64) *mat64.Dense {
	rows, cols := a.Dims()
	c := mat64.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			c.Set(i, j, f(a.At(i, j)))
		}
	}

	return c
}

func subtract(a *mat64.Dense, b *mat64.Dense) *mat64.Dense {
	rows, cols := a.Dims()
	c := mat64.NewDense(rows, cols, nil)

	c.Sub(a, b)
	return c
}

func scale(a *mat64.Dense, factor float64) *mat64.Dense {
	rows, cols := a.Dims()
	c := mat64.NewDense(rows, cols, nil)

	c.Scale(factor, a)
	return c
}
func multiply(a *mat64.Dense, b *mat64.Dense) *mat64.Dense {
	arows, acols := a.Dims()
	brows, bcols := b.Dims()

	if acols != brows {
		panic("matrix dimensions do not match for multiplication")
	}

	c := mat64.NewDense(arows, bcols, nil)

	c.Mul(a, b)

	return c
}
func scale2(s float64, m mat64.Matrix) mat64.Matrix {
	r, c := m.Dims()
	o := mat64.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}
func add2(m, n mat64.Matrix) mat64.Matrix {
	r, c := m.Dims()
	o := mat64.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}
func multiply2(m, n mat64.Matrix) mat64.Matrix {
	r, c := m.Dims()
	o := mat64.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}
func randomMatrix(r, c int) *mat64.Dense {
	result := mat64.NewDense(r, c, nil)
	randomize(result)
	return result
}

func randomize(m *mat64.Dense) {
	r, c := m.Dims()
	for row := 0; row < r; row++ {
		for col := 0; col < c; col++ {
			m.Set(row, col, rand.NormFloat64())
		}
	}
}

func sigmoidPrime(m *mat64.Dense) *mat64.Dense {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat64.NewDense(rows, 1, o)
	return multiply(m, subtract(ones, m)) // m * (1 - m)
}

func softmaxPrime(x mat64.Matrix) mat64.Matrix {
	r, c := x.Dims()
	y := mat64.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			y.Set(i, j, softmaxDerivFunc(x.At(i, j)))
		}
	}

	return y
}

// softmaxDerivFunc menghitung turunan dari softmax pada suatu nilai x
func softmaxDerivFunc(x float64) float64 {
	return softmaxFunc(x) * (1 - softmaxFunc(x))
}

// softmaxFunc menghitung nilai softmax pada suatu nilai x
func softmaxFunc(x float64) float64 {
	return goa.ClangOne(C.Exp, x) / (goa.ClangOne(C.Exp, x) + 1)
}

func expDivSumExp(m *mat64.Dense) *mat64.Dense {
	exp := &mat64.Dense{}
	exp.Apply(func(i, j int, v float64) float64 {
		return math.Exp(v)
	}, m)

	sumExp := mat64.Sum(exp)

	result := &mat64.Dense{}
	result.Apply(func(i, j int, v float64) float64 {
		return v / sumExp
	}, exp)

	return result
}

func tanhPrime(x mat64.Matrix) mat64.Matrix {
	r, c := x.Dims()
	y := mat64.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			y.Set(i, j, 1-pow(math.Tanh(x.At(i, j)), 2))
		}
	}

	return y
}

func CrossEntropyLoss(prediction mat64.Matrix, labels mat64.Matrix) float64 {
	var loss float64
	r, c := prediction.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			labelValue := labels.At(i, j)
			if labelValue == 0 {
				loss -= math.Log(1 - prediction.At(i, j) + 1e-9) // untuk mencegah pembagian dengan nol
			} else {
				loss -= math.Log(prediction.At(i, j) + 1e-9) // untuk mencegah pembagian dengan nol
			}
		}
	}
	loss /= float64(r)

	return loss
}
