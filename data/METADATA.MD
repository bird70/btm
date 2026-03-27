# Deakin Marine Mapping Group 2013-2014 Refuge Cove Dataset Metadata

## Location
- Site: Refuge Cove, Wilsons Promontory, Victoria, Australia
- Location: Approximately 39° 02′ 17.6″ S 146° 27′ 48.4″ E
- Depth range: 0-22 m

## Funding
- Parks Victoria

## License
- CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

## Remote sensing 

### Survey
- Type: Multibeam echosounder survey
- Vessel: Yolla RV
- Date: 11 June 2013
- Authors: Daniel Ierodiaconou, Alexandre Schimel, Sean Blake, Alex Rattray. DEAKIN UNIVERSITY
- Equipment: Kongsberg EM2040c Multibeam sonar. Valeport mini SVS. Applanix POS MV Wavemaster
- Frequency: 300 kHz
- Ping rate: Variable. Up to 50 Hz
- Pulse length: Variable. Down to 0.025 ms
- Beamforming mode: high-density equidistant. 400 soundings per ping
- Swath coverage: Constant. +- 65 degrees
- SVP: One profile at start. Valeport Monitor Sound Velocity Profiler
- GNSS solution: Differential. Fugro MarineStar
- Acquisition software: Kongsberg SIS

## Position processing
- Software: Applanix POSPac MMS 6.2
- Mode: IN-Fusion PPP
- Export: SBET in GDA94

### Bathymetry processing
- Software: CARIS HIPS&SIPS 8.1
- Heave: Delayed
- Attitude/navigation: SBET from Position processing
- Cleaning: Manual
- BASE surface: swath
- Resolution: 25cm
- Vertical reference: LAT
- Export:
  - geotif/RC_Z_25cm_UTM55S_p.tif: BASE surface exported as a 24Bit compressed tiff file.
  - xyz/RC_Z_25cm_GDA94_p.txt: gridded xyz exported in geographical coordinates 8dp precision, vertical reference GDA94 ellipsoid. 
  - xyz/RC_Z_25cm_UTM55S_LAT_p.txt: gridded xyz exported in UTM55S, vertical reference to LAT by shifting depth +0.75m (LAT height above GDA94 ellipsoid, from AusCoastVDT)

### Backscatter processing
- Software: IVS FMGT 7.3.5
- AVG: Trend 300 pings
- Sonar type: EM2040
- Mosaic settings: Blend 50%
- Resolution: 25cm
- Export:
  - geotif/RC_BS_25cm_UTM55S_p.tif: mosaic exported as a grayscale tif scaled between -40dB and -10dB.
  - fledermaus/RC_BS_25cm_UTM55S_p.sd: mosaic exported as a SD file

### Other final products
- Software: IVS Fledermaus 7.3.5
  - fledermaus/RC_Z_25cm_UTM55S_LAT_p.sd: imported RC_Z_25cm_UTM55S_LAT_p.txt, sign inverted.
  - fledermaus/RC_Zi_25cm_UTM55S_LAT_p.sd: bathy interpolated (3,5)
  - googleearth/RC_Zi_25cm_UTM55S_LAT_p.kmz: idem
  - fledermaus/RC_ZiBS_25cm_UTM55S_LAT_p.sd: mosaic drapped on interpolated bathy
  - googleearth/RC_ZiBS_25cm_UTM55S_LAT_p.kmz: idem
  - fledermaus/RC_Zicontours_25cm_UTM55S_LAT_p.sd: bathy contours at -2, -5, -10, -15, -20m
  - googleearth/RC_Zicontours_25cm_UTM55S_LAT_p.kml: idem
  - arcgis/Refuge_Cove.gdb: ArcGIS geodatabase containing the Z and BS layers

## Ground-truth

### Survey 1
- Type: AUV video transects
- Vehicle: OceanServer Technology Iver2-580 EP
- Date: 27-28 May 2013
- Authors: Australian Marine Ecology
- Sensor: GoPro Hero 3 Black video camera 
- Sensor setup: Camera mounted obliquely (45°) on the underside of AUV 
- Survey plan: 6 pre-programmed transects at a height of 1.2 m above the seabed and speed of 1.5 knots. 2 missions incomplete
- Position: DGPS position exported for every second of footage

### Survey 2
- Type: Drop video camera
- Date: 28 August 2014
- Authors: Sean Blake, Grace Gaylard, Daniel Ierodiaconou 
- Sensor: Delta vision HD underwater video camera
- Survey plan: 85 drops. Depth range 1.9 to 22.1 m
- Position: GPS raw data logged with Ashtec Mobile Mapper 10. Later improved to DGPS solution using VicMaps CORS base stations data

### Video processing
- Video interpretation:
  - Frequency: Every second for AUV video transect. Single interpretation for each drop video 
  - Scheme: All ground-truth sites assigned one of 5 broad habitat classes based on visual interpretation of the video footage:
     - Macroalgae Dominated Reef (ALG): High relief, granitic bedrock and boulder reef populated with diverse assemblages of brown, red and green macroalgal taxa varying in composition and density. Sessile invertebrates on vertical walls and in fissures.
    - Filamentous Mat (FMAT): Fine sandy to muddy sand sediments in sheltered and shallow area, covered in extensive mats of filamentous microalgae and diatoms. Interspersed with sparse to very sparse seagrass (Zostera sp.) shoots, often with evidence of bioturbation.
    - No Visible Biota (NVB): Extensive areas of fine to coarse sands and gravel with no visible epibiota.
    - Seagrass (Amphibolis antarctica) (SGAM): Predominantly dense beds of the seagrass A. antarctica, characterised by low, woody root masses at the sediment interface forming distinctive mounded edges surrounding the beds.
     - Seagrass (Zostera sp.) (SGZ): Beds of seagrass characterised by the presence of Zostera sp. growing in fine sands and muddy sands with patchy distribution ranging from sparse to dense. Where visible through the canopy, sediments frequently showed evidence of bioturbation, with mounds and burrows in sheltered areas.

## Metadata
- Authors: Alexandre Schimel, Grace Gaylard, Daniel Ierodiaconou
- Date: 1 Oct 2013, updated 7 March 2026