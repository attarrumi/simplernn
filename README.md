	// #cgo LDFLAGS: -lopenblas
	import "C"
	
	"github.com/gonum/blas/blas64"
	"github.com/gonum/blas/cgo"
    
    blas64.Use(cgo.Implementation{})
