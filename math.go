package simplernn

/*
#include <math.h>

double Tanh(double x) {
	return tanh(x);
}

double Pow(double x, double y) {
	return pow(x, y);
}
double Exp1(double x) {
	return exp(x);
}
*/
import "C"
import (
	"github.com/attarrumi/goa"
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + goa.ClangOne(C.Exp1, -x))
}
func tanh(z float64) float64 {
	return goa.ClangOne(C.Tanh, z)
}

func pow(x, y float64) float64 {
	return goa.ClangTwo(C.Pow, x, y)
}
