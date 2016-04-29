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

func Test2x2System2(t *testing.T) {
	system := &System{
		Size: 2,
		Coefficients: []DiffPoly{
			{2.0/25.0, 1}, {-1.0/50},
			{-2.0/25.0}, {2.0/25.0, 1},
		},
		Outputs: []NumFunc{zeroNumFunc, zeroNumFunc},
	}
	solution := &Solution{
		Funcs:        []DiffFunc{first2x2Solution2, second2x2Solution2},
		NumConstants: 2,
	}
	verification := Verify(system, solution, 1000, nil)
	if !verification[0] {
		t.Error("did not successfully verify equation 1")
	}
	if !verification[1] {
		t.Error("did not successfully verify equation 2")
	}
}

func Test3x3Solution(t *testing.T) {
	system := &System{
		Size: 3,
		Coefficients: []DiffPoly{
			{-1, 0, 1}, {2, 2}, {1, 1},
			{1, -2, 1}, {0, 4}, {-1, 2},
			{-1, 1}, {2}, {1, -1},
		},
		Outputs: []NumFunc{zeroNumFunc, zeroNumFunc, zeroNumFunc},
	}
	solution := &Solution{
		Funcs:        []DiffFunc{first3x3Solution, first2x2Solution, second2x2Solution},
		NumConstants: 4,
	}
	verification := Verify(system, solution, 1000, nil)
	if !verification[0] {
		t.Error("did not successfully verify equation 1")
	}
	if !verification[1] {
		t.Error("did not successfully verify equation 2")
	}
	if !verification[2] {
		t.Error("did not successfully verify equation 2")
	}

	badSolution := &Solution{
		Funcs:        []DiffFunc{second2x2Solution, first2x2Solution, first3x3Solution},
		NumConstants: 4,
	}
	verification = Verify(system, badSolution, 1000, nil)
	if verification[0] && verification[1] && verification[2] {
		t.Error("false positive")
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

func first3x3Solution(x *autodiff.DeepNum, consts []*autodiff.DeepNum) *autodiff.DeepNum {
	// k4*e^t - k2/2*t*e^-t + (3/4k2 + k3)e^-t + k1
	coeff1 := consts[3]
	coeff2 := consts[1].MulScaler(-1.0 / 2.0)
	coeff3 := consts[1].MulScaler(3.0 / 4.0).Add(consts[2])
	coeff4 := consts[0]

	term1 := x.Exp().Mul(coeff1)
	term2 := coeff2.Mul(x).Mul(x.MulScaler(-1).Exp())
	term3 := coeff3.Mul(x.MulScaler(-1).Exp())
	term4 := coeff4
	return term1.Add(term2).Add(term3).Add(term4)
}

func first2x2Solution2(x *autodiff.DeepNum, consts []*autodiff.DeepNum) *autodiff.DeepNum {
	// k1*exp(-1/25 x) + k2*exp(-3/25 x)
	exp1 := x.MulScaler(-1.0/25.0).Exp()
	exp2 := x.MulScaler(-3.0/25.0).Exp()
	return consts[0].Mul(exp1).Add(consts[1].Mul(exp2))
}

func second2x2Solution2(x *autodiff.DeepNum, consts []*autodiff.DeepNum) *autodiff.DeepNum {
	// 2*k1*exp(-1/25 x) - 2*k2*exp(-3/25 x)
	exp1 := x.MulScaler(-1.0/25.0).Exp().MulScaler(2)
	exp2 := x.MulScaler(-3.0/25.0).Exp().MulScaler(2)
	return consts[0].Mul(exp1).Sub(consts[1].Mul(exp2))
}

func zeroNumFunc(float64) float64 {
	return 0
}
