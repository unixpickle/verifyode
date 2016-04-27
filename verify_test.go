package verifyode

import (
	"testing"

	"github.com/unixpickle/num-analysis/autodiff"
)

func Test2x2System(t *testing.T) {
	system := &System{
		Size: 2,
		Coefficients: []DiffPoly{
			{}, {0, 1, 1},
			{2, 2}, {0, 0, 1},
		},
		Outputs: []NumFunc{zeroNumFunc, zeroNumFunc},
	}
	solution := &Solution{
		Funcs:        []DiffFunc{first2x2Solution, second2x2Solution},
		NumConstants: 3,
	}
	verification := Verify(system, solution, 1000, nil)
	if !verification[0] {
		t.Error("did not successfully verify equation 1")
	}
	if !verification[1] {
		t.Error("did not successfully verify equation 2")
	}

	badSolution := &Solution{
		Funcs:        []DiffFunc{second2x2Solution, first2x2Solution},
		NumConstants: 3,
	}
	verification = Verify(system, badSolution, 1000, nil)
	if verification[0] {
		t.Error("false positive for equation 1")
	}
	if verification[1] {
		t.Error("false positive for equation 2")
	}
}

func second2x2Solution(x *autodiff.DeepNum, consts []*autodiff.DeepNum) *autodiff.DeepNum {
	// k1 + k2*exp(-t)
	return consts[0].Add(consts[1].Mul(x.MulScaler(-1).Exp()))
}

func first2x2Solution(x *autodiff.DeepNum, consts []*autodiff.DeepNum) *autodiff.DeepNum {
	// -k2/2 * t * exp(-t) + k3*exp(-t)

	expInv := x.MulScaler(-1).Exp()
	term1 := consts[2].Mul(expInv)
	term2 := consts[1].MulScaler(-0.5).Mul(x).Mul(expInv)
	return term1.Add(term2)
}

func zeroNumFunc(float64) float64 {
	return 0
}
