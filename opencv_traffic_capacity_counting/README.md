# Road Traffic counting example based on OpenCV

**Speed:** 10.7 FPS (with visualization) 44.5 FPS (without visualization)

### Video visualization demo
[![Video visualization demo](https://img.youtube.com/vi/_o5iLbRHKao/0.jpg)](https://youtu.be/_o5iLbRHKao)

### Report example
![Report plot](report.png)

## Data
Go to http://keepvid.com/ and download video in 720p quality with url https://youtu.be/wqctLW0Hb_0

After running the script with defualt settings you will get **./out** dir with debug frames images and **report.csv** file with format "time, vehicles".

## How to run script
```
pip install -r ../requirements.txt
```

Edit **traffic_capacity.py** if needed:
```
IMAGE_DIR = "./out"
VIDEO_SOURCE = "input.mp4"
SHAPE = (720, 1280)  # HxW
AREA_PTS = np.array([
    [[732, 720], [732, 590], [1280, 500], [1280, 720]],
    [[0, 400], [645, 400], [645, 0], [0, 0]]
])

...

pipeline = PipelineRunner(pipeline=[
    CapacityCounter(area_mask=area_mask, image_dir=IMAGE_DIR),
    ContextCsvWriter('./report.csv',start_time=1505494325, fps=1, faster=10, field_names=['capacity']) # saving every 10 seconds
], log_level=logging.DEBUG)
```
Run script:
```
python traffic.py
```

## How to create video from processed images
```
chmod a+x make_video.sh
./make_video.sh
```

## How to create report plot
```
python plot.py [path to the csv report] [number of seconds to group by] 
```

