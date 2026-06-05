package strategy;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import org.apache.commons.statistics.descriptive.Mean;
import org.apache.commons.statistics.descriptive.StandardDeviation;
import org.ta4j.core.BarSeries;
import org.ta4j.core.BaseBarSeriesBuilder;
import org.ta4j.core.indicators.EMAIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;

import java.io.*;
import java.nio.file.*;
import java.time.*;
import java.time.format.*;
import java.util.*;
import java.util.stream.*;

public class StockBacktest {

    // -------------------------------------------------------------------------
    // CONFIG
    // -------------------------------------------------------------------------
    static final String DAILY_FILE  = "D:\\Stock_Strategy\\stock_data_daily.csv";
    static final String HOURLY_FILE = "D:\\Stock_Strategy\\input_stock_data_hourly.csv";
    static final String OUTPUT_FILE = "D:\\Stock_Strategy\\backtest_signals.csv";
    static final String LOG_FILE    = "D:\\Stock_Strategy\\backtest_logs.csv";
    static final String Z_SCORE_FILE    = "D:\\Stock_Strategy\\z_score_logs.csv";
    static final String CLOSE_FILE    = "D:\\Stock_Strategy\\monthly_close_logs.csv";
	static final String EMA_FILE = "D:\\Stock_Strategy\\hourly_EMA_logs.csv";
	static final String H_CLOSE_FILE = "D:\\Stock_Strategy\\H_close_logs.csv";
    

    static final int    LOOKBACK_DAYS     = 300;
    static final int    ZSCORE_WINDOW     = 100;
    static final double ZSCORE_THRESHOLD  = -.7;
    static final int    EMA_PERIOD        = 100;
    static final double EMA_THRESHOLD     = 0.90;

    public static void main(String[] args) throws Exception {

        // =========================================================================
        // PHASE 1: DATA LOADING
        // =========================================================================
        // Load daily OHLCV data from CSV.
        // Structure: ticker -> (field -> TimeSeries)
        // e.g., "RELIANCE" -> { "Close" -> TimeSeries{date->value}, "Open" -> ... }
        System.out.println("Loading daily data...");
        Map<String, Map<String, TimeSeries>> rawDaily = loadMultiHeaderCsv(DAILY_FILE);
        System.out.println("Daily timeseries size: " + rawDaily.size());

        // Load hourly OHLCV data from CSV — same structure as daily.
        // Used for intraday EMA computation and signal triggering.
        System.out.println("Loading hourly data...");
        Map<String, Map<String, TimeSeries>> rawHourly = loadMultiHeaderCsv(HOURLY_FILE);
        System.out.println("Hourly timeseries size: " + rawHourly.size());

        // Extract all ticker symbols found in the daily data.
        // These are the universe of stocks we will consider for backtesting.
        List<String> tickers = new ArrayList<>(rawDaily.keySet());

        // Master log: every ticker × every hourly bar gets one entry here.
        // Used later to write the full audit log CSV.
        List<LogEntry> allRowsLog = new ArrayList<>();


        // =========================================================================
        // PHASE 2: DATA VALIDATION
        // =========================================================================
        // Before computing anything, verify that each ticker has:
        //   - Sufficient daily history (at least ZSCORE_WINDOW bars for Z-score)
        //   - Sufficient hourly history (at least EMA_PERIOD bars for EMA)
        //   - No critical gaps or mismatches
        // Tickers failing validation are excluded from the backtest entirely.
        System.out.println("\nRunning data integrity and consistency tests...");
        List<String> validTickers = new ArrayList<>();

        for (String ticker : tickers) {
            ValidationResult vr = validateTicker(ticker, rawDaily, rawHourly,
                                                  ZSCORE_WINDOW, EMA_PERIOD);
            if (vr.passed) {
                // Ticker has clean, sufficient data — include in backtest
                validTickers.add(ticker);
            } else {
                // Ticker failed validation — log it and skip
                System.out.printf("  ⚠️  Ticker '%s' REJECTED | Reason: %s%n", ticker, vr.reason);

                // Add a one-time rejection entry into the master audit log
                // so we have a complete record of why this ticker was dropped
                LogEntry le = new LogEntry();
                le.dateTime    = "INITIALIZATION_PHASE";
                le.ticker      = ticker;
                le.rule1Passed = false;
                le.rule2Passed = false;
                le.status      = "REJECTED: " + vr.reason;
                allRowsLog.add(le);
            }
        }

        System.out.printf("-> Integrity checks complete. Proceeding with %d out of %d tickers.%n%n",
                validTickers.size(), tickers.size());

        // Replace original ticker list with only the validated set
        tickers = validTickers;


        // =========================================================================
        // PHASE 3: PRE-COMPUTATION — DAILY Z-SCORES
        // =========================================================================
        // Z-Score measures how many standard deviations the current close is
        // away from its rolling mean over ZSCORE_WINDOW days.
        //   Z = (Close - Mean) / StdDev
        // A Z-score ≤ ZSCORE_THRESHOLD (e.g., -1.4) indicates the stock is
        // statistically oversold — this is Rule 1 of our entry signal.
        //
        // Pre-computing and caching avoids recalculating on every hourly bar
        // during the backtest loop, which would be extremely slow.
        //
        // zscoreCache: ticker -> TimeSeries{ dailyDate -> zScoreValue }
        System.out.println("[Pre-computing] Daily Z-Scores...");
        Map<String, TimeSeries> zscoreCache = new HashMap<>();

        for (String ticker : tickers) {
            // Get the full daily close price history for this ticker
            TimeSeries closeDataAllDates = getClose(rawDaily, ticker);

            // Only compute Z-score if we have enough bars for the rolling window.
            // Without ZSCORE_WINDOW bars, the first Z-score values are unreliable.
            if (closeDataAllDates != null && closeDataAllDates.size() >= ZSCORE_WINDOW) {
                zscoreCache.put(ticker, calculateZScores(closeDataAllDates, ZSCORE_WINDOW));
            }
        }

        // Export the Z-score cache to CSV for offline inspection and debugging
        exportZscoreCacheToCsv(zscoreCache, Z_SCORE_FILE);
        System.out.println("Z Score logs written ...");


        // =========================================================================
        // PHASE 4: PRE-COMPUTATION —  CLOSING PRICE year 
        // =========================================================================

        System.out.println("[Pre-computing] Daily Last Month Closes (FIXED)...");
        Map<String, Map<LocalDateTime, Double>> yearCloseCache = new HashMap<>();

        rawDaily.forEach((ticker, fields) -> {
            TimeSeries closeSeries = fields.get("Close");

            if (closeSeries == null || closeSeries.isEmpty()) return; // skip if no close data

            Map<LocalDateTime, Double> rollingHighMap = new LinkedHashMap<>();

            closeSeries.sortedKeys().forEach(asOf ->
                closeSeries.highestClose(asOf, 12)
                           .ifPresent(high -> rollingHighMap.put(asOf, high))
            );

            yearCloseCache.put(ticker, rollingHighMap);
        });
        

		/*
		 * for (String ticker : tickers) { TimeSeries dailyClose = getClose(rawDaily,
		 * ticker); if (dailyClose != null && !dailyClose.isEmpty()) {
		 * yearCloseCache.put(ticker, computeYearlyClose(dailyClose)); } }
		 */

        
        // Export monthly close cache for audit / debugging purposes
        exportMonthlyCloseCacheToCsv(yearCloseCache, CLOSE_FILE);
        System.out.println("Close cache written ...");

        // =========================================================================
        // PHASE 5: PRE-COMPUTATION — HOURLY CLOSES AND EMA
        // =========================================================================
        // The 100-hour EMA (Exponential Moving Average) of the hourly close price
        // is the core momentum/trend indicator used in Rule 2.
        //   EMA gives more weight to recent prices vs. a simple moving average.
        //   A falling EMA below a threshold confirms a downtrend.
        //
        // hourlyCloseCache: ticker -> TimeSeries{ hourlyDateTime -> closePrice }
        // hourlyEmaCache  : ticker -> TimeSeries{ hourlyDateTime -> emaValue   }
        System.out.println("[Pre-computing] Hourly Closes & 100h EMAs...");
        Map<String, TimeSeries> hourlyCloseCache = new HashMap<>();
        Map<String, TimeSeries> hourlyEmaCache   = new HashMap<>();

        for (String ticker : tickers) {
            TimeSeries hClose = getClose(rawHourly, ticker);
            if (hClose != null && !hClose.isEmpty()) {
                hourlyCloseCache.put(ticker, hClose);

                // computeNewEma uses ta4j internally to calculate EMA
                // and returns results as our custom TimeSeries type
                hourlyEmaCache.put(ticker, computeNewEma(hClose, EMA_PERIOD));
            }
        }
        
        exportEmaCacheToCsv(hourlyEmaCache, EMA_FILE);
        exportEmaCacheToCsv(hourlyCloseCache, H_CLOSE_FILE);
        System.out.println("Hourly EMA cache written ...");


        // =========================================================================
        // PHASE 6: TIMELINE SETUP
        // =========================================================================
        // Build the master list of all unique trading days across all tickers.
        // Using a TreeSet ensures the dates are automatically sorted ascending.
        TreeSet<LocalDateTime> allTradingDaysSet = new TreeSet<>();
        for (String t : tickers) {
            TimeSeries ts = getClose(rawDaily, t);
            if (ts != null) allTradingDaysSet.addAll(ts.data.keySet());
        }
        List<LocalDateTime> allTradingDays = new ArrayList<>(allTradingDaysSet);

        // Determine the earliest date from which we have reliable Z-score values.
        // The first ZSCORE_WINDOW bars are the warm-up period — Z-scores computed
        // before this point use fewer samples and are statistically less reliable.
        LocalDateTime minHistoryDate = allTradingDays.size() > ZSCORE_WINDOW
                ? allTradingDays.get(ZSCORE_WINDOW)   // first bar after warm-up
                : allTradingDays.get(0);               // fallback if data is thin

        // Backtest window: run from (lastDay - LOOKBACK_DAYS) to last available day.
        // This focuses the analysis on the most recent relevant period.
        LocalDateTime lastDay    = allTradingDays.get(allTradingDays.size() - 1);
        LocalDateTime cutoffDate = lastDay.minusDays(LOOKBACK_DAYS);

        // startDate is the later of:
        //   (a) the lookback cutoff, or
        //   (b) the minimum date with reliable Z-score history
        // This prevents us from generating signals with incomplete indicator data.
        LocalDateTime startDate = cutoffDate.isAfter(minHistoryDate) ? cutoffDate : minHistoryDate;

        // Collect all unique hourly timestamps across all tickers,
        // filtered to only include those on or after the backtest startDate.
        // Each hourlyTs in this list will be one iteration in the backtest loop.
        TreeSet<LocalDateTime> allHourlySet = new TreeSet<>();
        for (String t : tickers) {
            TimeSeries ts = getClose(rawHourly, t);
            if (ts != null) allHourlySet.addAll(ts.data.keySet());
        }
        List<LocalDateTime> backtestHourly = allHourlySet.stream()
                .filter(dt -> !dt.isBefore(startDate))  // exclude pre-startDate bars
                .sorted()
                .collect(Collectors.toList());


        // =========================================================================
        // PHASE 7: BACKTEST LOOP — SIGNAL GENERATION
        // =========================================================================
        // This is the core simulation loop.
        // For every hourly bar × every valid ticker, we:
        //   1. Look up the most recent completed daily Z-score (no look-ahead bias)
        //   2. Look up the last month's closing price for that daily date
        //   3. Look up the current hourly EMA value
        //   4. Evaluate Rule 1 and Rule 2
        //   5. Classify and log the outcome
        //
        // Look-ahead bias prevention:
        //   At any given hourly bar on date D, we use daily data from date D-1
        //   (the last completed daily session before D). This mimics live trading
        //   where today's daily bar hasn't closed yet.
        System.out.printf("%nStarting backtest loop across %d hourly bars...%n", backtestHourly.size());
        List<SignalEntry> allSignals = new ArrayList<>();

        DateTimeFormatter dtFmt   = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm");
        DateTimeFormatter dateFmt = DateTimeFormatter.ofPattern("yyyy-MM-dd");

        for (LocalDateTime hourlyTs : backtestHourly) {

            // Strip time component to get the calendar date of this hourly bar
            LocalDateTime hourlyDate = hourlyTs.toLocalDate().atStartOfDay();
            String hourlyTsStr       = hourlyTs.format(dtFmt);

            // Look-ahead bias guard:
            // Find the last daily trading date STRICTLY BEFORE today's hourly date.
            // TreeSet.lower() returns the greatest element strictly less than the given value.
            // e.g., if hourlyTs is 2024-05-15 10:00, latestDailyDate = 2024-05-14 (or last trade day before that)
            LocalDateTime latestDailyDate = allTradingDaysSet.lower(hourlyDate);

            // If there's no prior trading day (e.g., we're at the very first bar),
            // skip this hourly bar entirely — we have no daily context yet
            if (latestDailyDate == null) continue;

            // Evaluate all tickers at this specific hourly timestamp
            for (String ticker : tickers) {

                // ------------------------------------------------------------------
                // INDICATOR LOOKUPS
                // ------------------------------------------------------------------

                // Rule 1 input: Z-Score from the last completed daily session
                // zScoreCache was keyed by daily dates, so we use latestDailyDate
                TimeSeries zSeries = zscoreCache.get(ticker);
                Double zscore = (zSeries != null) ? zSeries.get(latestDailyDate) : null;

                // Rule 2 input (part A): Last month's close as of the latest daily date
                // This forms the baseline price level for our threshold comparison
                Map<LocalDateTime, Double> lmcMap = yearCloseCache.get(ticker);
                Double lastMonthClose = (lmcMap != null) ? lmcMap.get(latestDailyDate) : null;

                // Rule 2 input (part B): Current hourly close price (informational, not used in rules directly)
                TimeSeries hCloseSeries = hourlyCloseCache.get(ticker);
                Double hourlyCloseVal   = (hCloseSeries != null) ? hCloseSeries.get(hourlyTs) : null;

                // Rule 2 input (part C): Hourly EMA value at this exact timestamp
                // This is compared against the price target derived from lastMonthClose
                TimeSeries hEmaSeries = hourlyEmaCache.get(ticker);
                Double hourlyEmaVal   = (hEmaSeries != null) ? hEmaSeries.get(hourlyTs) : null;

                // Compute the Rule 2 price target:
                // Target = EMA_THRESHOLD × lastMonthClose
                // e.g., if EMA_THRESHOLD = 0.90, the EMA must be below 90% of last month's close
                Double targetVal = (lastMonthClose != null) ? EMA_THRESHOLD * lastMonthClose : null;


                // ------------------------------------------------------------------
                // RULE EVALUATION
                // ------------------------------------------------------------------

                // RULE 1 — Mean Reversion / Oversold Condition (daily, Z-score based)
                // Passes when: Z-Score ≤ ZSCORE_THRESHOLD (e.g., -1.4)
                // Interpretation: Stock is statistically far below its historical mean,
                // suggesting an oversold condition and potential mean-reversion opportunity.
                boolean r1Passed = zscore != null && zscore <= ZSCORE_THRESHOLD;

                // RULE 2 — Trend Confirmation (hourly, EMA-based)
                // Passes when: Hourly EMA < EMA_THRESHOLD × Last Month's Close
                // Interpretation: The hourly trend (EMA) is still below a price-level threshold,
                // confirming that the stock hasn't already recovered before we entered.
                boolean r2Passed = hourlyEmaVal != null && targetVal != null
                                   && hourlyEmaVal < targetVal;


                // ------------------------------------------------------------------
                // STATUS CLASSIFICATION
                // ------------------------------------------------------------------
                // Each hourly bar gets exactly one of these status labels:
                //   MISSING_DATA  — one or more required indicators could not be resolved
                //   SIGNAL_MATCH  — both rules passed → actionable entry signal
                //   FAIL_RULE_2   — Rule 1 passed but Rule 2 failed (trend not confirmed)
                //   FAIL_RULE_1   — Rule 1 failed (stock not oversold enough)
                String status;
                if (zscore == null || lastMonthClose == null || hourlyEmaVal == null) {
                    // Cannot evaluate rules without all three indicators
                    status = "MISSING_DATA";
                } else if (r1Passed && r2Passed) {
                    // Both conditions met — this is a buy signal candidate
                    status = "SIGNAL_MATCH";
                } else if (r1Passed) {
                    // Stock is oversold (Rule 1 ✓) but EMA hasn't confirmed (Rule 2 ✗)
                    status = "FAIL_RULE_2";
                } else {
                    // Stock is not sufficiently oversold — primary filter failed
                    status = "FAIL_RULE_1";
                }


                // ------------------------------------------------------------------
                // AUDIT LOG ENTRY — one row per ticker per hourly bar
                // ------------------------------------------------------------------
                LogEntry le = new LogEntry();
                le.dateTime       = hourlyTsStr;
                le.ticker         = ticker;
                le.zScore         = round2(zscore, 4);
                le.rule1Passed    = r1Passed;
                le.hourlyClose    = round2(hourlyCloseVal, 2);
                le.lastMonthClose = round2(lastMonthClose, 2);
                le.target90pct    = round2(targetVal, 2);       // e.g., 90% of last month close
                le.ema100h        = round2(hourlyEmaVal, 2);
                le.rule2Passed    = r2Passed;
                le.status         = status;
                allRowsLog.add(le);

                // ------------------------------------------------------------------
                // SIGNAL CAPTURE — only for rows where both rules pass
                // ------------------------------------------------------------------
                if ("SIGNAL_MATCH".equals(status)) {
                    // Build a compact signal entry (subset of the full log entry)
                    // for the dedicated signals output CSV
                    SignalEntry se = new SignalEntry();
                    se.dateTime       = hourlyTsStr;
                    se.date           = hourlyDate.format(dateFmt);
                    se.ticker         = ticker;
                    se.zScore         = le.zScore;
                    se.lastMonthClose = le.lastMonthClose;
                    se.ema100h        = le.ema100h;
                    se.target90pct    = le.target90pct;
                    allSignals.add(se);
                }
            }
        }


        // =========================================================================
        // PHASE 8: OUTPUT — WRITE RESULTS TO CSV
        // =========================================================================
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Completed at: " + LocalDateTime.now().format(
                DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
        System.out.println("Writing files...");

        // Write the full audit log: every ticker × every hourly bar
        // Useful for debugging why a signal was or wasn't generated
        writeLogCsv(LOG_FILE, allRowsLog);
        System.out.printf("📊 Audit Log Matrix exported to: %s (%,d items matrixed)%n",
                LOG_FILE, allRowsLog.size());

        // Write only the matched signals (SIGNAL_MATCH rows) to a separate file
        // This is the primary actionable output of the backtest
        if (!allSignals.isEmpty()) {
            writeSignalCsv(OUTPUT_FILE, allSignals);
            System.out.printf("✅ Strategy Signals exported to: %s (%d records found)%n",
                    OUTPUT_FILE, allSignals.size());
        } else {
            // No signals found — either strategy too restrictive or data too thin
            System.out.println("⚠️  0 strategic matches tracked in the signals file.");
        }
    }
    
    static ValidationResult validateTicker(
            String ticker,
            Map<String, Map<String, TimeSeries>> rawDaily,
            Map<String, Map<String, TimeSeries>> rawHourly,
            int zWindow, int emaPeriod) {

        if (!rawDaily.containsKey(ticker))
            return new ValidationResult(false, "Missing entirely from Daily CSV columns");
        if (!rawHourly.containsKey(ticker))
            return new ValidationResult(false, "Missing entirely from Hourly CSV columns");

        try {
            TimeSeries dClose = getClose(rawDaily, ticker);
            TimeSeries hClose = getClose(rawHourly, ticker);

            if (dClose == null || dClose.size() < zWindow)
                return new ValidationResult(false,
                        String.format("Insufficient daily data (Has %d rows, needs %d for Z-Score)",
                                dClose == null ? 0 : dClose.size(), zWindow));

            if (hClose == null || hClose.size() < emaPeriod)
                return new ValidationResult(false,
                        String.format("Insufficient hourly data (Has %d rows, needs %d for EMA)",
                                hClose == null ? 0 : hClose.size(), emaPeriod));

            // Zero / negative check
            for (double v : dClose.data.values())
                if (v <= 0) return new ValidationResult(false,
                        "Data corruption: Contains zero or negative closing prices");
            for (double v : hClose.data.values())
                if (v <= 0) return new ValidationResult(false,
                        "Data corruption: Contains zero or negative closing prices");

            // Spike check (>400% in one day)
            List<LocalDateTime> dKeys = dClose.sortedKeys();
            for (int i = 1; i < dKeys.size(); i++) {
                double prev = dClose.get(dKeys.get(i - 1));
                double curr = dClose.get(dKeys.get(i));
                if (Math.abs(curr / prev - 1) > 4.0)
                    return new ValidationResult(false,
                            "Data anomaly: Contains an unrealistic price spike/drop (>400% in one day)");
            }

            // Timeline overlap check
            LocalDateTime dMin = dKeys.get(0);
            LocalDateTime dMax = dKeys.get(dKeys.size() - 1);
            List<LocalDateTime> hKeys = hClose.sortedKeys();
            LocalDateTime hMin = hKeys.get(0);
            LocalDateTime hMax = hKeys.get(hKeys.size() - 1);

            if (hMax.isBefore(dMin) || dMax.isBefore(hMin))
                return new ValidationResult(false,
                        String.format("Timeline Disconnect: Daily range (%s to %s) does not overlap Hourly range",
                                dMin.toLocalDate(), dMax.toLocalDate()));

        } catch (Exception e) {
            return new ValidationResult(false, "Unexpected data parsing error: " + e.getMessage());
        }

        return new ValidationResult(true, "Passed Integrity Check");
    }
    
    public static void exportEmaCacheToCsv(Map<String, TimeSeries> zscoreCache, String filePath) {
        DateTimeFormatter dateFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(filePath)))) {
              writer.println("Ticker,DateTime,ema");
            for (Map.Entry<String, TimeSeries> entry : zscoreCache.entrySet()) {
                String ticker = entry.getKey();
                TimeSeries timeSeries = entry.getValue();

                // Skip empty series
                if (timeSeries == null || timeSeries.isEmpty()) {
                    continue;
                }

                // 3. Iterate over the dates chronologically using your sortedKeys() method
                for (LocalDateTime dt : timeSeries.sortedKeys()) {
                    Double zScore = timeSeries.get(dt);
                    
                    String dateStr = dt.format(dateFormatter);
                    // Handle potential null values by writing an empty string (or change to "NaN" if preferred)
                    String zScoreStr = (zScore != null) ? String.valueOf(zScore) : "";

                    // 4. Write the row data
                    writer.println(ticker + "," + dateStr + "," + zScoreStr);
                }
            }
            
            System.out.println("✅ Z-Score data successfully exported to: " + filePath);

        } catch (IOException e) {
            System.err.println("❌ Error writing to CSV file: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public static void exportZscoreCacheToCsv(Map<String, TimeSeries> zscoreCache, String filePath) {
        DateTimeFormatter dateFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(filePath)))) {
              writer.println("Ticker,DateTime,ZScore");
            for (Map.Entry<String, TimeSeries> entry : zscoreCache.entrySet()) {
                String ticker = entry.getKey();
                TimeSeries timeSeries = entry.getValue();

                // Skip empty series
                if (timeSeries == null || timeSeries.isEmpty()) {
                    continue;
                }

                // 3. Iterate over the dates chronologically using your sortedKeys() method
                for (LocalDateTime dt : timeSeries.sortedKeys()) {
                    Double zScore = timeSeries.get(dt);
                    
                    String dateStr = dt.format(dateFormatter);
                    // Handle potential null values by writing an empty string (or change to "NaN" if preferred)
                    String zScoreStr = (zScore != null) ? String.valueOf(zScore) : "";

                    // 4. Write the row data
                    writer.println(ticker + "," + dateStr + "," + zScoreStr);
                }
            }
            
            System.out.println("✅ Z-Score data successfully exported to: " + filePath);

        } catch (IOException e) {
            System.err.println("❌ Error writing to CSV file: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void exportMonthlyCloseCacheToCsv(Map<String, Map<LocalDateTime, Double>> lastMonthCloseCache, String mCloseFile) {
        DateTimeFormatter dateFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(mCloseFile)))) {
              writer.println("Ticker,DateTime,Monthly_CLOSE");
            for (Map.Entry<String,  Map<LocalDateTime, Double>> entry : lastMonthCloseCache.entrySet()) {
                String ticker = entry.getKey();
                Map<LocalDateTime, Double> timeSeries = entry.getValue();

                // Skip empty series
                if (timeSeries == null || timeSeries.isEmpty()) {
                    continue;
                }

                // 3. Iterate over the dates chronologically using your sortedKeys() method
                for (LocalDateTime dt : timeSeries.keySet()) {
                    Double mClose = timeSeries.get(dt);
                    
                    String dateStr = dt.format(dateFormatter);
                    // Handle potential null values by writing an empty string (or change to "NaN" if preferred)
                    String mCloseStr = (mClose != null) ? String.valueOf(mClose) : "";

                    // 4. Write the row data
                    writer.println(ticker + "," + dateStr + "," + mCloseStr);
                }
            }
            
            System.out.println("✅ Monthly Close data successfully exported to: " + mCloseFile);

        } catch (IOException e) {
            System.err.println("❌ Error writing to CSV file: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    static Map<LocalDateTime, Double> computeYearlyClose(TimeSeries close) {
        Map<String, Double> yearlyMap = new LinkedHashMap<>();
        List<LocalDateTime> keys = close.sortedKeys();
        for (LocalDateTime dt : keys) {
            String key = dt.getYear() + "-" + dt.getMonthValue();
            yearlyMap.put(key, close.get(dt));   // last-write wins = last trading day
        }

        Map<LocalDateTime, Double> result = new LinkedHashMap<>();
        for (LocalDateTime date : keys) {
            int prevMonth = date.getMonthValue() == 1 ? 12 : date.getMonthValue() - 1;
            int prevYear  = date.getMonthValue() == 1 ? date.getYear() - 1 : date.getYear();
            String lookupKey = prevYear + "-" + prevMonth;
            result.put(date, yearlyMap.getOrDefault(lookupKey, null));
        }
        return result;
    }
    
    

    private static TimeSeries computeNewEma(TimeSeries hClose, int emaPeriod) {
    	BarSeries series = new BaseBarSeriesBuilder().withName("CloseOnlySeries").build();
        ZoneId systemZone = ZoneId.systemDefault();
        
     // Keep an ordered list of keys to map back bar index → LocalDateTime
        List<LocalDateTime> orderedKeys = new ArrayList<>();
        
        // Loop through your LinkedHashMap entries
        for (Map.Entry<LocalDateTime, Double> entry :  hClose.data.entrySet()) {
            // Convert LocalDateTime to ZonedDateTime required by ta4j
            ZonedDateTime zdt = entry.getKey().atZone(systemZone);
            double closePrice = entry.getValue();
            series.addBar(zdt, closePrice, closePrice, closePrice, closePrice, 0.0);
            orderedKeys.add(entry.getKey());
        }

    	
        ClosePriceIndicator closePriceIndicator = new ClosePriceIndicator(series);
        EMAIndicator emaIndicator = new EMAIndicator(closePriceIndicator, emaPeriod);
        TimeSeries result = new TimeSeries();
        for (int i = 0; i < series.getBarCount(); i++) {
            double emaValue = emaIndicator.getValue(i).doubleValue();
            result.put(orderedKeys.get(i), emaValue);
        }

		return result;
	}

	

	static Map<String, Map<String, TimeSeries>> loadMultiHeaderCsv(String path) throws Exception {

        Map<String, Map<String, TimeSeries>> result = new LinkedHashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {

            // --- Header row 0: tickers ---
            String line0 = br.readLine();
            String[] row0 = parseCsvLine(line0);   // col0 = blank / "Price"

            // --- Header row 1: fields ---
            String line1 = br.readLine();
            String[] row1 = parseCsvLine(line1);   // col0 = blank / "Ticker"

            // Build column-index -> (ticker, field) map
            // Column 0 is always the DateTime index column → skip
            List<String[]> colMeta = new ArrayList<>();   // index = col position (0-based incl. idx col)
            colMeta.add(new String[]{"__INDEX__", "__INDEX__"});
            for (int i = 1; i < row0.length; i++) {
                String ticker = (i < row0.length ? row0[i].trim() : "");
                String field  = (i < row1.length ? row1[i].trim() : "");
                colMeta.add(new String[]{ticker, field});
                result.computeIfAbsent(ticker, k -> new LinkedHashMap<>())
                      .computeIfAbsent(field,  k -> new TimeSeries());
            }

            // --- Data rows ---
            DateTimeFormatter[] fmts = {
                DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ssXXX"),
                DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"),
                DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"),
                DateTimeFormatter.ofPattern("yyyy-MM-dd")
            };

            String dataLine;
            while ((dataLine = br.readLine()) != null) {
                if (dataLine.isBlank()) continue;
                String[] cols = parseCsvLine(dataLine);

                LocalDateTime dt = parseDateTime(cols[0].trim(), fmts);
                if (dt == null) continue;

                for (int i = 1; i < cols.length && i < colMeta.size(); i++) {
                    String ticker = colMeta.get(i)[0];
                    String field  = colMeta.get(i)[1];
                    if (ticker.isEmpty() || field.isEmpty()) continue;

                    String raw = cols[i].trim();
                    if (raw.isEmpty() || raw.equalsIgnoreCase("nan")
                            || raw.equalsIgnoreCase("null")) continue;

                    try {
                        double val = Double.parseDouble(raw);
                        result.get(ticker).get(field).put(dt, val);
                    } catch (NumberFormatException ignored) { }
                }
            }
        }

        // Remove the placeholder keys used during construction (blank ticker / field names)
        result.remove("");
        return result;
    }

    

    // =========================================================================
    // INDICATOR COMPUTATIONS
    // =========================================================================

    /** Rolling Z-Score with the given window. */
    static TimeSeries calculateZScores(TimeSeries closeDataAllDates, int window) {    	
    	
        TimeSeries zScoreResults = new TimeSeries();
        
        //Keys is the list of all the dates in sorted format
        List<LocalDateTime> keys = closeDataAllDates.sortedKeys();
        
        if (closeDataAllDates == null || closeDataAllDates.size() < window) {
            return zScoreResults; // Return empty if there isn't enough data for even one window
        }
        

        
        for (int i = 0; i < keys.size(); i++) {
            if (i < window - 1) { continue;}
            
            int startIdx = i - window + 1;
            double[] windowData = new double[window];
            
            for (int j = 0; j < window; j++) {
            	windowData[j] = closeDataAllDates.get(keys.get(j+startIdx));
            }
            double sma = Mean.of(windowData).getAsDouble();
            double stdDev = StandardDeviation.of(windowData).getAsDouble();
            double currentClose = closeDataAllDates.get(keys.get(i));
            
            if (stdDev != 0) {
                double zScore = (currentClose - sma) / stdDev; // Important line
                zScoreResults.put(keys.get(i), zScore);
            } else {
                // If standard deviation is 0 (price hasn't moved for 200 days), Z-score is 0
                zScoreResults.put(keys.get(i), 0.0); 
            }
        }
        return zScoreResults;
    }

     /** Exponential Moving Average (EWM, adjust=False — matches pandas default). */
    static TimeSeries computeEma(TimeSeries close, int span) {
        TimeSeries result = new TimeSeries();
        double alpha = 2.0 / (span + 1);
        List<LocalDateTime> keys = close.sortedKeys();
        Double ema = null;
        for (LocalDateTime dt : keys) {
            Double price = close.get(dt);
            if (price == null) { result.put(dt, ema); continue; }
            ema = (ema == null) ? price : alpha * price + (1 - alpha) * ema;
            result.put(dt, ema);
        }
        return result;
    }

    // =========================================================================
    // CSV WRITERS
    // =========================================================================

    static void writeLogCsv(String path, List<LogEntry> rows) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println("DateTime,Ticker,Z_Score,Rule1_Passed,Hourly_Close," +
                       "Last_Month_Close,Target_90pct,EMA_100h,Rule2_Passed,Status");
            for (LogEntry r : rows) {
                pw.printf("%s,%s,%s,%b,%s,%s,%s,%s,%b,%s%n",
                        r.dateTime, r.ticker,
                        fmt(r.zScore), r.rule1Passed, fmt(r.hourlyClose),
                        fmt(r.lastMonthClose), fmt(r.target90pct), fmt(r.ema100h),
                        r.rule2Passed, r.status);
            }
        }
    }

    static void writeSignalCsv(String path, List<SignalEntry> rows) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(path))) {
            pw.println("DateTime,Date,Ticker,Z_Score,Last_Month_Close,EMA_100h,Target_90pct");
            for (SignalEntry r : rows) {
                pw.printf("%s,%s,%s,%s,%s,%s,%s%n",
                        r.dateTime, r.date, r.ticker,
                        fmt(r.zScore), fmt(r.lastMonthClose), fmt(r.ema100h), fmt(r.target90pct));
            }
        }
    }

    // =========================================================================
    // HELPERS
    // =========================================================================

    static TimeSeries getClose(Map<String, Map<String, TimeSeries>> raw, String ticker) {
        Map<String, TimeSeries> fields = raw.get(ticker);
        return fields == null ? null : fields.get("Close");
    }

    static Double round2(Double val, int places) {
        if (val == null || Double.isNaN(val)) return null;
        double factor = Math.pow(10, places);
        return Math.round(val * factor) / factor;
    }

    static String fmt(Double val) {
        return (val == null) ? "" : String.valueOf(val);
    }

    /** Minimal CSV line tokenizer (handles quoted fields). */
    static String[] parseCsvLine(String line) {
        List<String> tokens = new ArrayList<>();
        boolean inQuotes = false;
        StringBuilder sb = new StringBuilder();
        for (char c : line.toCharArray()) {
            if (c == '"') { inQuotes = !inQuotes; }
            else if (c == ',' && !inQuotes) { tokens.add(sb.toString()); sb.setLength(0); }
            else { sb.append(c); }
        }
        tokens.add(sb.toString());
        return tokens.toArray(new String[0]);
    }

    /** Try multiple datetime formats; return null if none match. */
    static LocalDateTime parseDateTime(String raw, DateTimeFormatter[] fmts) {
        // Strip timezone suffix if present (we tz_localize(None) in Python)
        String s = raw.replaceAll("[+-]\\d{2}:\\d{2}$", "").trim();
        for (DateTimeFormatter fmt : fmts) {
            try {
                // If the format only covers a date, convert to midnight LocalDateTime
                if (fmt.toString().contains("HH")) {
                    return LocalDateTime.parse(s, fmt);
                } else {
                    return LocalDate.parse(s, fmt).atStartOfDay();
                }
            } catch (Exception ignored) { }
        }
        return null;
    }
}
