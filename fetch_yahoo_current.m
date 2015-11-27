function data = fetch_yahoo_current(symbol)
  url = 'http://download.finance.yahoo.com/d/quotes.csv?s=';
  suffix = '&f=ohgl1v&e=.csv';
  full_url = strcat(url, symbol, suffix);
  data_str = urlread(full_url);
  [open, high, low, close, volume] = strread(data_str, "%f,%f,%f,%f,%f");
  data = [open, high, low, close, volume];
end
