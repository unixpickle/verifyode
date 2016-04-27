package verifyode

import (
	"math"
	"math/rand"

	"github.com/unixpickle/num-analysis/autodiff"
	"github.com/unixpickle/num-analysis/kahan"
)

const defaultMaxError = 1e-11

// Scale specifies appropriate magnitudes
// for test values for a system of ODEs.
type Scale struct {
	ConstMagnitude []float64
	VarMagnitude   float64

	// MaxError is the maximum acceptable error for
	// a solution.
	// Error is considered absolutely, so it should
	// be chosen wisely.
	// This is due to the fact that many systems
	// of ODEs have outputs of 0, and that nothing
	// can be relatively compared to 0.
	MaxError float64
}

// Verify verifies a solution to a system of
// ODEs by running a solution on a bunch of
// random inputs.
// It will run count test cases, and it will
// return an array of bools, where each bool
// corresponds to an equation in the system.
// If an equation was satisfied, its corresponding
// entry in the returned []bool will be set.
//
// If Scale is nil, all of the constants will be
// bounded by a magnitude of 1.
func Verify(sys *System, sol *Solution, count int, scale *Scale) []bool {
	if scale == nil {
		scale = defaultScale(sol.NumConstants)
	}

	res := make([]bool, sys.Size)
	for i := range res {
		res[i] = true
	}

	depth := sys.Depth()

	for i := 0; i < count; i++ {
		input := (rand.Float64()*2 - 1) * scale.VarMagnitude
		consts := make([]*autodiff.DeepNum, sol.NumConstants)
		for i := range consts {
			mag := scale.ConstMagnitude[i]
			c := (rand.Float64()*2 - 1) * mag
			consts[i] = autodiff.NewDeepNum(c, depth)
		}
		inputVal := autodiff.NewDeepNumVar(input, depth)
		varValues := make([]*autodiff.DeepNum, len(sol.Funcs))
		for i, f := range sol.Funcs {
			varValues[i] = f(inputVal, consts)
		}
		actual := actualOutputs(sys, varValues)
		expected := expectedOutputs(sys, input)
		for i, x := range expected {
			a := actual[i]
			if math.IsNaN(a) || math.Abs(a-x) > scale.MaxError {
				res[i] = false
			}
		}
	}
	return res
}

func defaultScale(numConsts int) *Scale {
	scale := &Scale{
		VarMagnitude:   1,
		ConstMagnitude: make([]float64, numConsts),
		MaxError:       defaultMaxError,
	}
	for i := range scale.ConstMagnitude {
		scale.ConstMagnitude[i] = 1
	}
	return scale
}

func actualOutputs(s *System, varValues []*autodiff.DeepNum) []float64 {
	if len(varValues) != s.Size {
		panic("system's size does not match number of variables")
	}
	res := make([]float64, s.Size)
	coeffIdx := 0
	for i := range res {
		output := kahan.NewSummer64()
		for j := 0; j < s.Size; j++ {
			variable := varValues[j]
			diffPoly := s.Coefficients[coeffIdx]
			coeffIdx++

			for _, coefficient := range diffPoly {
				output.Add(variable.Value * coefficient)
				variable = variable.Deriv
			}
		}
		res[i] = output.Sum()
	}
	return res
}

func expectedOutputs(s *System, x float64) []float64 {
	res := make([]float64, s.Size)
	for i, f := range s.Outputs {
		res[i] = f(x)
	}
	return res
}
