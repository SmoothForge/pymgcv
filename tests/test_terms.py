from dataclasses import dataclass

import pytest

from pymgcv.basis_functions import CubicSpline, PSpline, ThinPlateSpline
from pymgcv.terms import AbstractTerm, Interaction, Offset, S, T
from pymgcv.terms import L as L


@dataclass
class TermTestCase:
    term: AbstractTerm
    expected_str: str
    expected_simple: str
    expected_simple_with_idx: str


test_cases = [
    TermTestCase(
        term=L("a"),
        expected_str="a",
        expected_simple="a",
        expected_simple_with_idx="a.1",
    ),
    TermTestCase(
        term=Interaction("a", "b", "c"),
        expected_str="a:b:c",
        expected_simple="a:b:c",
        expected_simple_with_idx="a:b:c.1",
    ),
    TermTestCase(
        term=S("a"),
        expected_str="s(a)",
        expected_simple="s(a)",
        expected_simple_with_idx="s.1(a)",
    ),
    TermTestCase(
        term=S("a", by="b"),
        expected_str="s(a,by=b)",
        expected_simple="s(a):b",
        expected_simple_with_idx="s.1(a):b",
    ),
    TermTestCase(
        term=S("a", "b", bs=ThinPlateSpline(m=3)),
        expected_str='s(a,b,bs="tp",m=3L)',
        expected_simple="s(a,b)",
        expected_simple_with_idx="s.1(a,b)",
    ),
    TermTestCase(
        term=S(
            "a",
            "b",
            k=10,
            bs=CubicSpline(cyclic=True),
            by="var",
            id=2,
            fx=True,
        ),
        expected_str='s(a,b,by=var,k=10L,bs="cc",id=2L,fx=TRUE)',
        expected_simple="s(a,b):var",
        expected_simple_with_idx="s.1(a,b):var",
    ),
    TermTestCase(
        term=S("a", bs=PSpline(cyclic=True)),
        expected_str='s(a,bs="cp",m=c(2L, 2L))',
        expected_simple="s(a)",
        expected_simple_with_idx="s.1(a)",
    ),
    TermTestCase(
        term=T("a", "b"),
        expected_str="te(a,b)",
        expected_simple="te(a,b)",
        expected_simple_with_idx="te.1(a,b)",
    ),
    TermTestCase(
        term=T("a", "b", interaction_only=True),
        expected_str="ti(a,b)",
        expected_simple="ti(a,b)",
        expected_simple_with_idx="ti.1(a,b)",
    ),
    TermTestCase(
        term=T(
            "x1",
            "x2",
            "x3",
            bs=[ThinPlateSpline(m=2), CubicSpline()],
            d=[2, 1],
            by="var",
            np=False,
            id=2,
            fx=True,
            interaction_only=True,
        ),
        expected_str='ti(x1,x2,x3,by=var,bs=c("tp", "cr"),m=list(2L, NA),d=2:1,id=2L,fx=TRUE,np=FALSE)',
        expected_simple="ti(x1,x2,x3):var",
        expected_simple_with_idx="ti.1(x1,x2,x3):var",
    ),
    TermTestCase(
        term=T("x", "y", bs=[CubicSpline(), CubicSpline()]),
        expected_str='te(x,y,bs=c("cr", "cr"))',
        expected_simple="te(x,y)",
        expected_simple_with_idx="te.1(x,y)",
    ),
]


@pytest.mark.parametrize("test_case", test_cases)
def test_smooth_to_str(test_case: TermTestCase):
    assert test_case.expected_str == str(test_case.term)
    assert test_case.expected_simple == test_case.term.mgcv_identifier()
    assert test_case.expected_simple_with_idx == test_case.term.mgcv_identifier(1)


def test_term_addition():
    l0 = L("x0")
    l1 = S("x1")
    l2 = T("x2", "x3")
    expected = [l0, l1, l2]
    assert l0 + l1 + l2 == expected
    assert expected + Offset("x3") == expected + [Offset("x3")]
    assert Interaction("x3", "x4") + expected == [Interaction("x3", "x4")] + expected
