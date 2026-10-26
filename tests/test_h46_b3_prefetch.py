"""Synthetic memory admission checks; no molecular computation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from h46_b3_prefetch import headroom,memory_estimate


def test_future_main_growth_stays_reserved():
    assert headroom(available=80,forecast=90,live_rss=30,auxiliary_reserve=5)==15


def test_no_backfill_when_future_peak_uses_free_memory():
    assert headroom(available=50,forecast=90,live_rss=30,auxiliary_reserve=0)==0


def test_main_rss_above_forecast_does_not_increase_available_memory():
    assert headroom(available=50,forecast=20,live_rss=30,auxiliary_reserve=5)==45


def test_main_forecast_requires_a_completed_comparable_size():
    row=dict(system='H6',s=32768)
    small=dict(candidate=dict(system='H6',s=1024),peak_RSS_bytes=2**30)
    slopes={'H6':900000}
    assert memory_estimate(row,slopes,[small])==2**30+900000*32768


def test_fresh_large_trial_refreshes_forecast_with_growth_margin():
    row=dict(system='H6',s=32768)
    measured=dict(candidate=dict(system='H6',s=16384),peak_RSS_bytes=8*2**30)
    assert memory_estimate(row,{'H6':900000},[measured])==2**30+1.25*(8*2**30-512*2**20)*2


def test_auxiliary_peak_does_not_remove_per_event_growth():
    small=dict(candidate=dict(system='H4',s=32),peak_RSS_bytes=600*2**20)
    a=memory_estimate(dict(system='H4',s=32),{'H4':300000},[small],auxiliary=True)
    b=memory_estimate(dict(system='H4',s=1024),{'H4':300000},[small],auxiliary=True)
    assert b-a==300000*(1024-32)


def test_observed_peak_above_original_forecast_increases_reservation():
    row=dict(system='H6',s=32768)
    measured=dict(candidate=dict(system='H6',s=32768),peak_RSS_bytes=40*2**30)
    assert memory_estimate(row,{'H6':900000},[measured])>40*2**30
