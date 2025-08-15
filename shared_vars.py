from datetime import datetime

loader = None

OPT_PATH='../MARKET_DATA/deribit_options_snapshots/BTC_snapshots_1m.csv'
FUT_PATH='../MARKET_DATA/_crypto_data/BTCUSDT/BTCUSDT_1m.csv'
START=datetime(2023, 1, 1)
END=datetime(2025, 1, 1)

# Источник — один огромный файл:
SRC_SNAPSHOT_CSV = "../MARKET_DATA/deribit_options_snapshots/BTC_snapshots_1m.csv"

# Куда писать помесячные файлы:
OUT_MONTH_DIR = "../MARKET_DATA/deribit_options_snapshots/by_month/BTC"

# Шаблон имени помесячного файла (будет подставлен currency, year, month)
FILENAME_TEMPLATE = "{currency}_snapshots_1m_{year:04d}-{month:02d}.csv"

# Путь к манифесту (создаётся при сплите и затем используется загрузчиком)
MANIFEST_PATH = "../MARKET_DATA/deribit_options_snapshots/by_month/manifest.json"

data_fut = None

positions_list = []

total = 0

metrics = {}