import asyncio
import shared_vars as sv
import data.fut_loader as load_f

from data.opt_loader import MonthlySnapshotLoader
import settings as set
import loop.run as r
import vizualizer.viz as viz
import statistic.stat as stat
import vizualizer.correlation as cor


async def main():
    #=======LOAD DATA=========
    sv.data_fut = load_f.load_candles(path=sv.FUT_PATH, start_date=sv.START, end_date=sv.END)
    if set.with_opt:
        sv.loader = MonthlySnapshotLoader(month_dir=sv.OUT_MONTH_DIR, manifest_path=sv.MANIFEST_PATH, currency="BTC")
    #=========================
    
    await r.run_loop()
    
    viz.plot_profit_bars_with_stats(sv.positions_list, out_dir="_viz_statistic")
    st = stat.compute_trading_stats(sv.positions_list)
    stat.plot_stats_overview(st, out_dir="_viz_statistic", filename="overview_stats.png")
    cor.analyze_trades(sv.positions_list, metric_keys=list(sv.metrics.keys()))
    #cor.visualize_joint(sv.positions_list, out_dir="_viz_statistic", n_bins_x=6, n_bins_y=6, min_count=1, metric_keys=list(sv.metrics.keys()))


if __name__ == "__main__":
    asyncio.run(main())