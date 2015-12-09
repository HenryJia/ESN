function data = fetch_yahoo_historical(date, Symbol, All = false)
  #fetch(connection , 'AAL.L', date='27-Nov-2015')
  pkg load financial

  connection = yahoo();

  if (All = true)
    Symbols = char(textread("symbols_clean_yahoo.csv", '%s'));
    data = zeros(size(Symbols, 1), 5);
    for i = 1:size(Symbols, 1)
      data(i, :) = fetch(connection , Symbols(i, :), date=date)(2:6);
    end
  end
  else
    data = fetch(connection , Symbol, date=date)(2:6);
  end
end