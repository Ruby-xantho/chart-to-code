'''
Connect to the Binance API using the python-binance library.
'''



from binance.client import Client
import os

# Option 1: Set API credentials as environment variables
# export BINANCE_API_KEY='api_key'
# export BINANCE_API_SECRET='api_secret'

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

# Option 2: Hardcode credentials (not recommended for production)
# api_key = 'api_key'
# api_secret = 'api_secret'

# Initialize the Binance Client
client = Client(api_key, api_secret)

# Example: Get account information
try:
    account_info = client.get_account()
    print("Account Information:")
    print(account_info)
except Exception as e:
    print(f"Error fetching account info: {e}")

# Example: Get current ticker prices
try:
    prices = client.get_all_tickers()
    print("Current Ticker Prices (first 5):")
    for ticker in prices[:5]:
        print(f"{ticker['symbol']}: {ticker['price']}")
except Exception as e:
    print(f"Error fetching ticker prices: {e}")

# Example: Place a test order (TEST endpoint)
try:
    order = client.create_test_order(
        symbol='BTCUSDT',
        side=Client.SIDE_BUY,
        type=Client.ORDER_TYPE_LIMIT,
        timeInForce=Client.TIME_IN_FORCE_GTC,
        quantity=0.001,
        price='30000'
    )
    print("Test order placed successfully (no actual order executed)")
except Exception as e:
    print(f"Error with test order: {e}")
