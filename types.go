package verifyode

import "github.com/unixpickle/num-analysis/autodiff"

// DiffPoly represents a polynomial of
// differentiation operations.
//
// The n-th element in a DiffPoly corresponds
// to the coefficient for the operator D^n.
//
// For example, DiffPoly{1, 2, -2, 1} represents
// the operator (1+2D-2D^2+D^3), where D is
// differentiation.
type DiffPoly []float64

// DiffFunc is an infinitely differentiable
// function of one variable.
//
// A DiffFunc also takes an array of constants,
// which are not considered variables since the
// function is not differentiated with respect
// to them.
//
// A DiffFunc must output a DeepNum which is
// precisely as deep as its argument, t.
type DiffFunc func(t *autodiff.DeepNum, constants []*autodiff.DeepNum) *autodiff.DeepNum

// NumFunc is a function of one numerical
// variable which needn't be differentiable.
type NumFunc func(f float64) float64

// A System represents a system of ODEs.
// A given NxN system has N unknowns (all
// DiffFuncs), N equations, and N outputs.
// Each output is itself a function, and
// is the result of a given equation.
//
// For example, consider the system:
//
//     (D-1)x + (D+1)y = e^x
//     (D^2)x +     4y = 2e^(-x)
//
// In said system, x and y are the unknowns,
// (D-1), (D+1), (D^2), and 4 are coefficients,
// and e^x and 2e^(-x) are outputs.
type System struct {
	// Size is the number of equations and unknowns.
	Size int

	// Coefficients is an unrolled list of
	// coefficients in the system.
	// It goes from left to right, then top
	// to bottom.
	Coefficients []DiffPoly

	// Outputs are the outputs of the system
	// starting from the output of the first
	// equation and going down.
	Outputs []NumFunc
}

// Depth returns the maximum number of
// derivatives that any coefficient takes.
// This is 0 if the system takes no derivatives.
func (s *System) Depth() int {
	var res int
	for _, c := range s.Coefficients {
		if len(c)-1 > res {
			res = len(c) - 1
		}
	}
	return res
}

// A Solution is a set of functions which
// supposedly solve an ODE.
// A given solution may also have unbounded
// constants which can be set to anything.
type Solution struct {
	Funcs        []DiffFunc
	NumConstants int
}
