 -- Aggregation horaire par pickup_community_area
CREATE OR REPLACE TABLE `avisia-certification-ml-yde.chicago_taxis.demand_by_hour` AS
SELECT
  TIMESTAMP_TRUNC(trip_start_timestamp, HOUR) AS timestamp_hour,
  pickup_community_area,
  COUNT(*) AS trip_count
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE trip_start_timestamp BETWEEN '2023-01-01' AND '2023-12-31'
  AND pickup_community_area IS NOT NULL
GROUP BY timestamp_hour, pickup_community_area
ORDER BY timestamp_hour, pickup_community_area;
