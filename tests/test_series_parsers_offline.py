"""Offline tests for the XML point parser. They need no API key."""
import pandas as pd
import pytest

from entsoe.series_parsers import _extract_timeseries, _parse_timeseries_generic


def _xml(resolution, start, end, points, curve='A01', label='quantity'):
    body = ''.join(
        f'<Point><position>{pos}</position><{label}>{val}</{label}></Point>' for pos, val in points)
    return (
        '<Doc><TimeSeries><curveType>%s</curveType><Period>'
        '<timeInterval><start>%s</start><end>%s</end></timeInterval>'
        '<resolution>%s</resolution>%s</Period></TimeSeries></Doc>' % (curve, start, end, resolution, body))


def _parse(xml, **kwargs):
    soup = next(_extract_timeseries(xml))
    return _parse_timeseries_generic(soup, merge_series=True, **kwargs)


def test_hourly_points_and_thousands_separator():
    xml = _xml('PT60M', '2024-03-30T23:00Z', '2024-03-31T02:00Z', [(1, '1,000.5'), (2, '2'), (3, '3')])
    series = _parse(xml)
    assert list(series.values) == [1000.5, 2.0, 3.0]
    assert list(series.index) == list(pd.date_range('2024-03-30T23:00Z', periods=3, freq='h'))


def test_quarter_hour_points():
    xml = _xml('PT15M', '2025-10-01T00:00Z', '2025-10-01T01:00Z', [(i, i) for i in range(1, 5)])
    series = _parse(xml)
    assert list(series.index) == list(pd.date_range('2025-10-01T00:00Z', periods=4, freq='15min'))
    assert list(series.values) == [1.0, 2.0, 3.0, 4.0]


def test_a03_missing_positions_are_forward_filled():
    xml = _xml('PT60M', '2024-01-01T00:00Z', '2024-01-01T05:00Z', [(1, 10), (4, 40)], curve='A03')
    series = _parse(xml)
    assert list(series.values) == [10.0, 10.0, 10.0, 40.0, 40.0]


def test_price_label_and_unsorted_positions():
    xml = _xml('PT60M', '2024-01-01T00:00Z', '2024-01-01T03:00Z', [(3, 30), (1, 10), (2, 20)], label='price.amount')
    series = _parse(xml, label='price.amount')
    assert list(series.values) == [10.0, 20.0, 30.0]


def test_daily_resolution_keeps_calendar_offsets():
    xml = _xml('P1D', '2024-03-30T23:00Z', '2024-04-02T23:00Z', [(1, 1), (2, 2)])
    series = _parse(xml)
    assert list(series.index) == [pd.Timestamp('2024-03-30T23:00Z'), pd.Timestamp('2024-03-31T23:00Z')]


@pytest.mark.parametrize('n', [1, 96])
def test_many_points_have_unique_sorted_index(n):
    xml = _xml('PT15M', '2024-01-01T00:00Z', '2024-01-02T00:00Z', [(i, i) for i in range(1, n + 1)])
    series = _parse(xml)
    assert series.index.is_monotonic_increasing and series.index.is_unique and len(series) == n


def test_repeated_position_keeps_last_value():
    xml = _xml('PT60M', '2024-01-01T00:00Z', '2024-01-01T02:00Z', [(1, '10'), (1, '11'), (2, '12')])
    series = _parse(xml)
    assert list(series.values) == [11.0, 12.0]
    assert series.index.is_unique
