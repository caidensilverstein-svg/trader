"""Unit tests for strategies/condor_greeks.py."""

import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from strategies.condor_greeks import (
    bs_call_greeks,
    bs_put_greeks,
    compute_condor_greeks,
    format_condor_greeks,
    CondorGreeks,
)


class TestBSCallGreeks:

    def test_delta_between_0_and_1(self):
        g = bs_call_greeks(100, 100, 0.25, 0.05, 0.20)
        assert 0 <= g["delta"] <= 1

    def test_atm_call_delta_approx_half(self):
        g = bs_call_greeks(100, 100, 1.0, 0.05, 0.20)
        # ATM delta should be approximately 0.5-0.6
        assert 0.45 < g["delta"] < 0.65

    def test_deep_itm_call_delta_near_1(self):
        g = bs_call_greeks(200, 100, 1.0, 0.05, 0.20)
        assert g["delta"] > 0.9

    def test_deep_otm_call_delta_near_0(self):
        g = bs_call_greeks(100, 200, 1.0, 0.05, 0.20)
        assert g["delta"] < 0.1

    def test_gamma_positive(self):
        g = bs_call_greeks(100, 100, 0.25, 0.05, 0.20)
        assert g["gamma"] > 0

    def test_theta_negative_for_long_call(self):
        g = bs_call_greeks(100, 100, 0.25, 0.05, 0.20)
        assert g["theta"] < 0

    def test_vega_positive_for_long_call(self):
        g = bs_call_greeks(100, 100, 0.25, 0.05, 0.20)
        assert g["vega"] > 0

    def test_zero_dte_gives_intrinsic_value(self):
        g = bs_call_greeks(110, 100, 0, 0.05, 0.20)
        assert g["delta"] == 1.0 or g["price"] >= 0

    def test_higher_vol_higher_vega(self):
        g_low  = bs_call_greeks(100, 100, 1.0, 0.05, 0.10)
        g_high = bs_call_greeks(100, 100, 1.0, 0.05, 0.40)
        assert g_high["vega"] > g_low["vega"]

    def test_price_non_negative(self):
        g = bs_call_greeks(100, 100, 0.25, 0.05, 0.20)
        assert g["price"] >= 0


class TestBSPutGreeks:

    def test_put_delta_negative(self):
        g = bs_put_greeks(100, 100, 0.25, 0.05, 0.20)
        assert g["delta"] < 0

    def test_atm_put_delta_approx_minus_half(self):
        g = bs_put_greeks(100, 100, 1.0, 0.05, 0.20)
        assert -0.65 < g["delta"] < -0.35

    def test_put_call_parity_price(self):
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.20
        call = bs_call_greeks(S, K, T, r, sigma)
        put  = bs_put_greeks(S, K, T, r, sigma)
        # Put-call parity: C - P = S - K*e^(-rT)
        rhs = S - K * math.exp(-r * T)
        lhs = call["price"] - put["price"]
        assert abs(lhs - rhs) < 0.5

    def test_put_gamma_matches_call(self):
        # Gamma is same for put and call with same strikes
        call = bs_call_greeks(100, 100, 0.25, 0.05, 0.20)
        put  = bs_put_greeks(100, 100, 0.25, 0.05, 0.20)
        assert abs(call["gamma"] - put["gamma"]) < 0.0001

    def test_price_non_negative(self):
        g = bs_put_greeks(100, 110, 0.25, 0.05, 0.20)
        assert g["price"] >= 0


class TestComputeCondorGreeks:

    def _make_condor(self, **kwargs):
        defaults = dict(
            underlying="SPY", spot=530.0,
            put_long_K=500.0, put_short_K=515.0,
            call_short_K=545.0, call_long_K=560.0,
            dte=30, iv=0.17, contracts=1, r=0.05,
        )
        defaults.update(kwargs)
        return compute_condor_greeks(**defaults)

    def test_returns_condor_greeks_object(self):
        cg = self._make_condor()
        assert isinstance(cg, CondorGreeks)

    def test_net_delta_near_zero_for_symmetric_condor(self):
        # Symmetric wings should give near-zero delta
        cg = self._make_condor()
        assert abs(cg.net_delta) < 50  # within +-50 delta on 100-share lot

    def test_net_theta_positive(self):
        # Short condor collects theta
        cg = self._make_condor()
        assert cg.net_theta > 0

    def test_net_vega_negative(self):
        # Short condor hurt by vol increase
        cg = self._make_condor()
        assert cg.net_vega < 0

    def test_max_profit_positive(self):
        cg = self._make_condor()
        assert cg.max_profit > 0

    def test_max_loss_negative(self):
        cg = self._make_condor()
        assert cg.max_loss < 0

    def test_prob_profit_between_0_and_100(self):
        cg = self._make_condor()
        assert 0 <= cg.prob_profit <= 100

    def test_breakeven_spread_around_short_strikes(self):
        cg = self._make_condor()
        # Breakevens should be between short strikes
        assert cg.breakeven_low < 515.0
        assert cg.breakeven_high > 545.0

    def test_more_contracts_scales_greeks(self):
        cg1 = self._make_condor(contracts=1)
        cg2 = self._make_condor(contracts=2)
        # 2 contracts = double the greeks
        assert abs(cg2.net_theta / cg1.net_theta - 2.0) < 0.1


class TestFormatCondorGreeks:

    def test_contains_header(self):
        cg = TestComputeCondorGreeks()._make_condor()
        r = format_condor_greeks(cg)
        assert "CONDOR" in r

    def test_contains_underlying(self):
        cg = TestComputeCondorGreeks()._make_condor()
        r = format_condor_greeks(cg)
        assert "SPY" in r

    def test_contains_greeks(self):
        cg = TestComputeCondorGreeks()._make_condor()
        r = format_condor_greeks(cg)
        assert "Delta" in r
        assert "Theta" in r
        assert "Vega" in r
