import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.gcp.bigquery import WriteToBigQuery
from datetime import datetime
import logging

class ProcessTaxiTrip(beam.DoFn):
    def process(self, element):
        """Process each taxi trip record."""
        try:
            # Convert string timestamps to datetime objects
            trip_start = datetime.strptime(element['trip_start_timestamp'], '%Y-%m-%d %H:%M:%S UTC')
            trip_end = datetime.strptime(element['trip_end_timestamp'], '%Y-%m-%d %H:%M:%S UTC')
            
            # Calculate trip duration in minutes
            trip_duration = (trip_end - trip_start).total_seconds() / 60
            
            # Extract temporal features
            hour_of_day = trip_start.hour
            day_of_week = trip_start.weekday()
            month = trip_start.month
            
            # Calculate speed (miles per hour)
            speed = element['trip_miles'] / (trip_duration / 60) if trip_duration > 0 else 0
            
            # Create processed record
            processed_record = {
                'trip_id': element['trip_id'],
                'trip_start_timestamp': element['trip_start_timestamp'],
                'trip_end_timestamp': element['trip_end_timestamp'],
                'trip_miles': element['trip_miles'],
                'pickup_latitude': element['pickup_latitude'],
                'pickup_longitude': element['pickup_longitude'],
                'dropoff_latitude': element['dropoff_latitude'],
                'dropoff_longitude': element['dropoff_longitude'],
                'trip_duration': trip_duration,
                'hour_of_day': hour_of_day,
                'day_of_week': day_of_week,
                'month': month,
                'speed': speed
            }
            
            yield processed_record
            
        except Exception as e:
            logging.error(f"Error processing record: {e}")
            yield None

def run():
    """Main function to run the Dataflow pipeline."""
    # Pipeline options
    options = PipelineOptions(
        project='avisia-certification-ml-yde',
        region='europe-west1',
        runner='DataflowRunner',
        temp_location='gs://avisia-certification-ml-yde-temp/temp',
        staging_location='gs://avisia-certification-ml-yde-temp/staging',
        save_main_session=True
    )
    
    # Define the pipeline
    with beam.Pipeline(options=options) as p:
        # Read from BigQuery
        query = """
        SELECT
            trip_id,
            trip_start_timestamp,
            trip_end_timestamp,
            trip_miles,
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude
        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
        WHERE trip_start_timestamp IS NOT NULL
            AND trip_end_timestamp IS NOT NULL
            AND trip_miles > 0
            AND pickup_latitude IS NOT NULL
            AND pickup_longitude IS NOT NULL
            AND dropoff_latitude IS NOT NULL
            AND dropoff_longitude IS NOT NULL
        """
        trips = (p
                 | 'Read from BigQuery' >> beam.io.Read(beam.io.BigQuerySource(query=query, use_standard_sql=True))
                )
        
        # Process trips
        processed_trips = (trips
                           | 'Process Trips' >> beam.ParDo(ProcessTaxiTrip())
                           )
        
        # Write to BigQuery
        processed_trips | 'Write to BigQuery' >> WriteToBigQuery(
            table='avisia-certification-ml-yde:chicago_taxis.processed_trips',
            schema='trip_id:STRING,trip_start_timestamp:TIMESTAMP,trip_end_timestamp:TIMESTAMP,'
                   'trip_miles:FLOAT,pickup_latitude:FLOAT,pickup_longitude:FLOAT,'
                   'dropoff_latitude:FLOAT,dropoff_longitude:FLOAT,trip_duration:FLOAT,'
                   'hour_of_day:INTEGER,day_of_week:INTEGER,month:INTEGER,speed:FLOAT',
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE
        )

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
