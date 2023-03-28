package simplernn

/*
#include <math.h>

double Tanh(double x) {
	return tanh(x);
}

double Pow(double x, double y) {
	return pow(x, y);
}
*/
import "C"
import (
	"github.com/attarrumi/goa"
)

func tanh(z float64) float64 {
	return goa.ClangOne(C.Tanh, z)
}

func pow(x, y float64) float64 {
	return goa.ClangTwo(C.Pow, x, y)
}
