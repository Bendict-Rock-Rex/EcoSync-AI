This readme.txt file was generated on 2021-10-05 by Tianzhen Hong of LBNL

GENERAL INFORMATION

1. Title of Dataset: LBNL Building 59: A three-year dataset supporting research on building energy, HVAC controls, and occupancy analytics

2. Author Information
	A. Principal Investigator Contact Information
		Name: Tianzhen Hong
		Institution: Lawrence Berkeley National Laboratory
		Address: 1 Cyclotron Road, Berkeley, CA 94720 USA
		Email: thong@lbl.gov

	B. Associate or Co-investigator Contact Information
		Name: Na Luo
		Institution: Lawrence Berkeley National Laboratory
		Address: 1 Cyclotron Road, Berkeley, CA 94720 USA
		Email: nluo@lbl.gov

	C. Alternate Contact Information
		Name: David Blum
		Institution: Lawrence Berkeley National Laboratory
		Address: 1 Cyclotron Road, Berkeley, CA 94720 USA
		Email: DHBlum@lbl.gov

3. Date of data collection (single date, range, approximate date): 2018-01-01 to 2020-12-31

4. Geographic location of data collection: Berkeley, California, USA

5. Information about funding sources that supported the collection of the data: The data collection was supported by the model predictive control project, which was part of the U.S.-China joint Clean Energy Research Center on Building Energy Efficiency (CERC-BEE) 2016-2021. The data processing effort was part of the Benchmark Datasets project. Both projects are supported by the Assistant Secretary for Energy Efficiency and Renewable Energy, Office of Building Technologies of the United States Department of Energy, under Contract No. DE-AC02-05CH11231. 


SHARING/ACCESS INFORMATION

1. Licenses/restrictions placed on the data: N/A

2. Links to publications that cite or use the data: N/A

3. Links to other publicly accessible locations of the data: https://bbd.labworks.org/ds/bbd/lbnlbldg59

4. Links/relationships to ancillary data sets: N/A

5. Was data derived from another source? No 

6. Recommended citation for this dataset:  N. Luo, Z. Wang, D. Blum, N. Bourassa, J. Broughton, C. Weyandt,  M.A. Piette, T. Hong. A three-year dataset supporting research on building energy, HVAC controls, and occupancy analytics. submitted to Nature Scientific Data, under review.


DATA & FILE OVERVIEW

1. File List: 
	- Bldg59_clean data: This folder contains 27 time-series data files in CSV format, covering energy use data, indoor and outdoor environmental data, HVAC control and operational data, as well as occupant-related data. The data has been processed using the data cleaning script (https://github.com/LBNL-ETA/Data-Cleaning) to fill the gap of the raw data.
	- Bldg59_w_occ Brick model.ttl: This TTL file is the Brick model of the dataset, which represents the metadata and hierarchical structure of the building, systems and sensors.
	- Bldg59_w_occ metadata of dataset.json: This JSON file represents the high-level building information, data governance, contextual information for each data category, as well as application perspectives of the dataset. 

2. Relationship between files, if important: First, users can obtain the high-level metadata about the building and dataset in the metadata JSON file. Then, we recommend users to explore the metadata of equipment and sensors in the Brick model by using the Brick TTL viewer. Finally, the time-series data is in CSV format and has a size of 2.38 GB.

3. Additional related data collected that was not included in the current data package: N/A

4. Are there multiple versions of the dataset? No


METHODOLOGICAL INFORMATION

1. Description of methods used for collection/generation of data: 
Data of Building 59 come from various sources and systems. Some key examples are as follows: 
	- For the operational data of the HVAC systems, we use the ALC SOAP web interface to retrieve data of specified points from ALC WebCTRL Building Automation System (BAS).
	- For the electrical consumption data, we query the ElasticSearch database on which the data is held for specific points through a web endpoint. 
	- For the site weather data, we make HTTP RESTful requests to the Synopticlabs web API to gain access to data from the weather station located on LBNL campus.
	- For occupant data, we deployed camera-based occupancy sensors, manufactured by the TRAF-SYS company, to collect occupant count data. 
	- For indoor temperature data, we deployed 16 air temperature sensors built with Raspberry Pi Zero W and DS18B20 Digital Temperature Sensors.
Finally, to store the data in a central database for ease of use, all the data streams are pulled from their sources and systems and integrated into an influxdb database, an open source time series database. 

2. Methods for processing the data: 
We identified and modified the data gaps and outlier values to generate the clean version of the time-series data. Specifically, we used multiple algorithms (e.g., linear interpolation, K-nearest neighbors, Matrix Factorization) to fill the gaps by considering both the length of data gap and the sampling frequency of each data point. More detaild description can be found at: https://github.com/LBNL-ETA/Data-Cleaning.

3. Instrument- or software-specific information needed to interpret the data: 
- The time-series data is stored in CSV format, which can be further interpreted in any software/tool (e.g., Microsoft Excel, code-based interpreters).
- The Brick model is stored in TTL format, which can be interpreted using the Brick TTL Viewer. 
- The metadata JSON file can be interpreted using any text readers. 

4. Standards and calibration information, if appropriate: N/A

5. Environmental/experimental conditions: The data were collected from an office building under real operating conditions.

6. Describe any quality-assurance procedures performed on the data: We cleaned the dataset by filling the data gaps and adjusting the outliers.

7. People involved with sample collection, processing, analysis and/or submission: Na Luo, Zhe Wang, David Blum, Norman Bourassa, Jeffrey Broughton, Chris Weyandt, Mary Ann Piette, Tianzhen Hong, all from Lawrence Berkeley National Laboratory. 


DATA-SPECIFIC INFORMATION FOR: [FILENAME]
<repeat this section for each dataset, folder or file, as appropriate>

We have 27 time-series data files in CSV format. We summarized the following information for each file in a seperate excel file, titled as "data description table_3year_clean data.xlsx". 

1. Number of variables: 

2. Number of cases/rows: 

3. Variable List: 
<list variable name(s), description(s), unit(s)and value labels as appropriate for each>

4. Missing data codes: N/A

5. Specialized formats or other abbreviations used: 
