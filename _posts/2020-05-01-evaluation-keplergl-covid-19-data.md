---
layout: post
title:  "An evaluation of Kepler.gl for Covid-19-style geographic timeseries"
date:   2020-05-01 14:00:00 +0200
categories: blog
permalink: evaluation-keplergl-covid-19-data
math: false
---
I'm a big fan of visualizing how things change over time on maps. Previously, this had me plotting ugly maps with matplotlib-based tooling, writing jpegs to a filesystem and combining them to a video or gif with `ffmpeg`. Not an entirely pleasant experience. When I first realised that Uber's [Kepler.gl](https://kepler.gl/) not only looks great, but has built-in time series support as well, it got me pretty excited!

I will evaluate Kepler.gl when used with geographic time series data. In particular, with the kind of daily data, aggregated per region, that we see a lot during the Covid-19 epidemic. The dataset that I'll use was made in a simulation of Covid-19 infections in neighbourhoods around Schiphol in The Netherlands, made earlier in [this post](https://jvlanalytics.nl/covid-19-simulation).

**Update 2020-05-02:** I decided to also create a version of [the circle map (Map 1) with real worldwide data](/assets/blog/2020-05-01-evaluation-keplergl-covid-19-data/covid-19-map.html), rather than demoing only simulated data. I used [this data source](https://github.com/CSSEGISandData/COVID-19/). As I've explained below, try not to change the size of the time range, as that will just result in circles being drawn on top of each other. Refresh to reset it.

```python
from datetime import date, timedelta
import json
from pathlib import Path

from IPython.display import HTML
from keplergl import KeplerGl
import pandas as pd
import geopandas as gpd

DATA_PATH = Path("../covid-19-simulation/data")
SIM_NAME = "SimulationResult(n_days=250,tr_day=50,sd_day=40)"
SIM_RES_PATH = DATA_PATH / "results" / "sim-30km" / SIM_NAME
 # First infection in The Netherlands minus incubation time:
START_DATE = date(2020, 2, 27) - timedelta(days=5) 
```

# Loading and prepping data
Okay, we have some data munging work to do first. Feel free to skip this and scroll down for the mapping goodies.

### Load population & geo data
This is the same data as used in [my previous post](https://jvlanalytics.nl/covid-19-simulation), taken from Statistics Netherlands. I'm using `GeoPandas` to load it. The `wkt` format for polygons is a string that Kepler.gl understands. These polygons are of neighbourhoods in The Netherlands. We also have the population numbers, which will be useful for normalizing data.


```python
df_pop = (gpd
          .read_file(DATA_PATH / "transformed" / "population.shp")
          .assign(centroid=lambda df: df["geometry"].map(lambda geo: 
                                                         geo.centroid))
          .assign(lon=lambda df: df.centroid.map(lambda c: c.x),
                  lat=lambda df: df.centroid.map(lambda c: c.y),
                  geo_wkt=lambda df: df["geometry"].map(lambda geo: 
                                                        geo.to_wkt())
          )
          # "hood" is a engineering abbrevation here, rather than slang ;-)
          [["hood", "muni", "pop", "lon", "lat", "geo_wkt"]]
)
df_pop.index.name = "hood_id"
df_pop.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hood</th>
      <th>muni</th>
      <th>pop</th>
      <th>lon</th>
      <th>lat</th>
      <th>geo_wkt</th>
    </tr>
    <tr>
      <th>hood_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wijk 02 Belgisch Park</td>
      <td>'s-Gravenhage</td>
      <td>7955</td>
      <td>4.292010</td>
      <td>52.112078</td>
      <td>POLYGON ((4.2874663832711484 52.11844854220218...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Wijk 03 Westbroekpark en Duttendel</td>
      <td>'s-Gravenhage</td>
      <td>1855</td>
      <td>4.303211</td>
      <td>52.104034</td>
      <td>POLYGON ((4.3000017436689992 52.09899194050742...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Wijk 04 Benoordenhout</td>
      <td>'s-Gravenhage</td>
      <td>13320</td>
      <td>4.321339</td>
      <td>52.097413</td>
      <td>POLYGON ((4.3273890785303832 52.09565737879255...</td>
    </tr>
  </tbody>
</table>
</div>



### Load simulation data
We have simulated counts of susceptible, infected, recovered and deceased persons, per neighbourhood, per day. 


```python
daily_count_dfs = []

for day, path_daily_count_csv in \
        enumerate(sorted(SIM_RES_PATH.glob("daily_counts*"))):
    date = START_DATE + timedelta(days=day)

    df_dc = (
        pd.read_csv(path_daily_count_csv)
        .assign(date=date)
        .set_index(["date", "hood_id"])
        .rename(columns={"PersonState.infected": "infected"})
        [["infected"]]
        .applymap(int)
    )
    
    daily_count_dfs.append(df_dc)
    
df_daily_counts = (
    pd.concat(daily_count_dfs)
    .sort_index()
    # Create a string version of the date for display purposes:
    .assign(date_str=lambda df: df
            .index.get_level_values("date").strftime("%Y-%m-%d"))
)
```


```python
pd.concat((df_daily_counts.head(2), df_daily_counts.tail(2)))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>infected</th>
      <th>date_str</th>
    </tr>
    <tr>
      <th>date</th>
      <th>hood_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2020-02-22</th>
      <th>85</th>
      <td>0</td>
      <td>2020-02-22</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0</td>
      <td>2020-02-22</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2020-10-28</th>
      <th>2740</th>
      <td>44</td>
      <td>2020-10-28</td>
    </tr>
    <tr>
      <th>2741</th>
      <td>6</td>
      <td>2020-10-28</td>
    </tr>
  </tbody>
</table>
</div>



### Join the two datasets

I haven't found a way to join two datasets in Kepler.gl. Which is a shame when using time series data with static polygons, because these relatively large polygons have to duplicated for every timestep. This - as we shall see below - blows up the dataset to such an extent that Kepler refuses to work with it.


```python
df_daily_counts_geo = df_daily_counts.merge(df_pop, left_index=True, right_index=True, how="inner")
df_daily_counts_geo[["infected", "hood", "pop", "lon", "lat", "geo_wkt"]].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>infected</th>
      <th>hood</th>
      <th>pop</th>
      <th>lon</th>
      <th>lat</th>
      <th>geo_wkt</th>
    </tr>
    <tr>
      <th>date</th>
      <th>hood_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">2020-02-22</th>
      <th>85</th>
      <td>0</td>
      <td>Wijk 00 Aalsmeer</td>
      <td>12030</td>
      <td>4.736807</td>
      <td>52.253185</td>
      <td>POLYGON ((4.7437654333167867 52.26983760839533...</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0</td>
      <td>Wijk 01 Kudelstraat en Kalslagen</td>
      <td>9200</td>
      <td>4.739852</td>
      <td>52.237020</td>
      <td>POLYGON ((4.7550087572913968 52.25162737834186...</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0</td>
      <td>Wijk 02 Oosteinde</td>
      <td>9840</td>
      <td>4.795393</td>
      <td>52.282207</td>
      <td>POLYGON ((4.7680638414593837 52.27023804377086...</td>
    </tr>
  </tbody>
</table>
</div>



### Create normalized counts
It's all too easy to create maps that simply reflect the population density. A good way to compensate for that is to normalize: divide by the total population count.


```python
df_daily_counts_geo["infected_percentage"] = \
    (df_daily_counts_geo["infected"] / 
     df_daily_counts_geo["pop"] * 100).round(1)
```

We also need to do a bit of hacking in order to make Kepler normalize across the entire time series. This actually happens by default in the first two maps I'll show below, but not in the next two maps that follow (for understandable reasons). I'm using a small dummy triangle in the North Sea, outside of the visible area, with the relevant maxima:

```python
uniq_dates = df_daily_counts_geo.index.get_level_values("date").unique()
df_dummy = pd.DataFrame(data={
    "date": uniq_dates,
    "date_str": uniq_dates.strftime("%Y-%m-%d"),
    "hood_id": -1,
    "hood": "dummy",
    "muni": "dummy",
    "pop": 0,
    "infected": df_daily_counts_geo["infected"].max(),
    "infected_percentage": df_daily_counts_geo["infected_percentage"].max(),
    "lon": 3.3267975,
    "lat": 52.3911063,
    "geo_wkt": "POLYGON ((3.3267975 52.3911063, 3.3535767 52.3638604, "
               "3.3851624 52.3915253))",
}).set_index(["date", "hood_id"])

df = pd.concat((df_daily_counts_geo, df_dummy)).sort_index()
```

# Mapping with Kepler.gl
Let's fire up Kepler. The config can be stored in a JSON file and passed into the `KeplerGl(..)` constructor as seen below. Modifications are best made in the GUI, after which the JSON can be stored in order to re-create the map later.  It has pretty sensible default behavior by automatically parsing the `lon`/`lat`/`geo_wkt` columns, but some tweaking is usually required still.

### Map 1: Circles like the famous John Hopkins Covid-19 map
I'll first attempt to re-create the [John Hopkins Covid-19 map](https://gisanddata.maps.arcgis.com/apps/opsdashboard/index.html?fbclid=IwAR0Q7KKPfPT3uhysJlRi8fTpqNzHkMkd6NOWDYA7tkDYuSFCuHs85Z2e-uw#/bda7594740fd40299423467b48e9ecf6), which shows cumulative counts of reported infections. The underlying data is updated on a daily basis. It's not very detailed for The Netherlands, unfortunately: 

![John Hopkins map Europe](/assets/blog/2020-05-01-evaluation-keplergl-covid-19-data/img/john_hopkins.png "John Hopkins map Europe")

This version will be zoomed in to a 30km radius around Schiphol.


```python
KEPLER_CONF_CIRCLES = "kepler-config-circles.json"
COLS_CIRCLES = ["infected", "infected_percentage", "hood", "muni", "lon", "lat"]

with open(KEPLER_CONF_CIRCLES, "r") as fp:
    config = json.load(fp)

map_circles = KeplerGl(
    height=800, 
    data={"data": df[COLS_CIRCLES].reset_index()},
    config=config
)
map_circles
```


    KeplerGl(config={'version': 'v1', 'config': {'visState': {'filters': [{'dataId': 'data', 'id': 'x8ikqz08r', 'n…


    User Guide: https://github.com/keplergl/kepler.gl/blob/master/docs/keplergl-jupyter/user-guide.md



```python
# Make sure to run this after making changes in the GUI (if they need to be persisted):
with open(KEPLER_CONF_CIRCLES, "w") as fp:
    json.dump(map_circles.config, fp)
```

Screenshot:

![Circles map](/assets/blog/2020-05-01-evaluation-keplergl-covid-19-data/img/map_circles.png "Circles map")    

The timeline feature is activated here. While it looks good, I am already facing some significant usability issues:

- Kepler.gl does not work very well with data that's already aggregated per coordinate and per time interval. If, for example, the time range is 2 days, 2 circles will be drawn on top of each other.
- Hence, I want the time range to be exactly 1 day. This isn't doable in the GUI, but we can edit the JSON file manually (see `timeRange`, fill in Unix timestamps such that the initial time is from 00:00:00 to 23:59:59 on the same day).
- But now, with the time range indicator being exactly 1 day, it becomes too small to select. It's not possible anymore to navigate through the timeline by hand. I'm stuck with using the play & pause buttons.
- The bottom graph doesn't show anything useful, even when setting the y-axis.

Kepler.gl seems to have been built for unaggregated data that is generated at a random moment in time. Which isn't a complete surprise, since this exactly describes the nature of Uber rides!

Ideally, I'd use daily *new* cases instead of cumulative. If only we could inform Kepler to aggregate (sum) circles on the exact same coordinates instead of drawing them on top of each other, this would be a fantastic way to easily identify areas with a lot of growth in a particular time range of arbitrary length. 

But, I have to admit, the playback feature is very nice:


```python
HTML('<video controls loop><source src="/assets/blog/2020-05-01-evaluation-keplergl-covid-19-data/vid/vid_circles_v3.mp4"></video>')
```




<video controls loop><source src="/assets/blog/2020-05-01-evaluation-keplergl-covid-19-data/vid/vid_circles_v3.mp4"></video>



### Map 2: Choropleth map
Choropleth maps are a common sight, but they can be tricky to interpret. For instance, they overemphasize the importance of large, potentially sparsely inhabited areas. Also, they hide differences within the regions and they may give the false impression of abrupt change at borders. 

But, it just so happens that my population data is on the level of neighbourhoods, and taking the centroid and placing a circle there doesn't do it justice entirely either. Let's see what it looks like with neighbourhood polygons. This time, I'll use the normalized infection numbers which allows us to identify areas that are more heavily affected, relatively speaking.

Unfortunately, it seems impossible to load *all* data. As mentioned earlier, Kepler.gl requires data to be passed in a denormalized form. The Polygons are huge, and with the constraint of having to include them for every timestep, it seems I'm hitting some dataset size limit. It results in a Python stacktrace (`tornado.iostream.StreamClosedError`) and a JavaScript error in the browser console. This might just work when using Kepler.gl directly rather than through Python, but personally I'm much more interested in the Python API than the JavaScript API. Fortunately, if we limit the dataset to just 2 points per month, it works:


```python
date_range = pd.date_range(start=df.date_str.min(), 
                           end=df.date_str.max(), freq="15D")
```


```python
KEPLER_CONF_CHORO_DYNAMIC = "kepler-config-choro-dynamic.json"
COLS_CHORO = ["infected_percentage", "hood", "muni", "geo_wkt", "date_str"]

with open(KEPLER_CONF_CHORO_DYNAMIC, "r") as fp:
    config = json.load(fp)

map_choro_dynamic = KeplerGl(
    height=800, 
    data={"data": df[COLS_CHORO].loc[date_range, :].reset_index()},
    config=config
)

map_choro_dynamic
```

    User Guide: https://github.com/keplergl/kepler.gl/blob/master/docs/keplergl-jupyter/user-guide.md



    KeplerGl(config={'version': 'v1', 'config': {'visState': {'filters': [{'dataId': 'data', 'id': 'y5phjeri9', 'n…



```python
with open(KEPLER_CONF_CHORO_DYNAMIC, "w") as fp:
    json.dump(map_choro_dynamic.config, fp)
```

Screenshot:

![Choropleth map](/assets/blog/2020-05-01-evaluation-keplergl-covid-19-data/img/map_choro_dynamic.png "Choropleth map")


```python
HTML('<video controls loop><source src="/assets/blog/2020-05-01-evaluation-keplergl-covid-19-data/vid/vid_choro_v2.mp4"></video>')
```




<video controls loop><source src="/assets/blog/2020-05-01-evaluation-keplergl-covid-19-data/vid/vid_choro_v2.mp4"></video>



As before, the key to getting the transitions *just right* is to edit the JSON file manually and change the unix timestamps in the `timeRange` field to be exactly equal to the interval of the data. The GUI doesn't allow for sufficiently precise control. If the `timeRange` is slightly too small, the data disappears shortly. If the window is slightly too large, the two days in the window will be drawn on top of each other (which visually comes across as "flickering" due to the transparency).

Now, there _is_ actually a hacky way to get daily data, but that involves taking the time control out of Kepler.gl and into the Python kernel. This is where we start needing the trick to normalize across the entire time series, because we are only feeding in 1 day at a time:


```python
KEPLER_CONF_CHORO_STATIC = "kepler-config-choro-static.json"
COLS_CHORO = ["infected_percentage", "hood", "muni", "geo_wkt", "date_str"]

with open(KEPLER_CONF_CHORO_STATIC, "r") as fp:
    config = json.load(fp)

map_choro_static = KeplerGl(
    height=800, 
    data={"data": df[COLS_CHORO].loc[date_range.min()].reset_index()},
    config=config
)

map_choro_static
```

    User Guide: https://github.com/keplergl/kepler.gl/blob/master/docs/keplergl-jupyter/user-guide.md



    KeplerGl(config={'version': 'v1', 'config': {'visState': {'filters': [], 'layers': [{'id': 'y08tsf', 'type': '…



```python
with open(KEPLER_CONF_CHORO_STATIC, "w") as fp:
    json.dump(map_choro_static.config, fp)
```


```python
for date in df.index.get_level_values("date").unique():
    new_data = df[COLS_CHORO].loc[date].reset_index()
    map_choro_static.add_data(new_data, name="data")
```

We'd need to tune a `time.sleep(...)` call to slow it down if needed. I'm not doing it here: I'm happy with the speed I'm getting on my computer.


```python
HTML('<video controls loop><source src="/assets/blog/2020-05-01-evaluation-keplergl-covid-19-data/vid/vid_choro_workaround_v2.mp4"></video>')
```




<video controls loop><source src="/assets/blog/2020-05-01-evaluation-keplergl-covid-19-data/vid/vid_choro_workaround_v2.mp4"></video>



Due to not having the timeline controls, we now have to use the tooltip to see the current date. There's also no way to pause this easily or to go back and forward in time. Hence, this workaround is mostly useful for making videos (which, then, do allow pause and scrolling backwards and forwards, but obviously at the expense of other useful interactive things such as tooltips).

### Map 4: 3D histogram with hexbins
Data Scientists love their histograms. The hexbin feature approximates a 3D histogram and looks pretty fancy. This one also requires the "normalization across the entire time series" trick, probably because it would be quite complex to pre-compute all possible bins over the entire timeline.


```python
KEPLER_CONF_HEXBIN = "kepler-config-hexbin.json"
COLS_HEXBIN = ["infected", "lon", "lat"]

with open(KEPLER_CONF_HEXBIN, "r") as fp:
    config = json.load(fp)

map_hexbin = KeplerGl(
    height=800, 
    data={"data": df[COLS_HEXBIN].reset_index()},
    config=config
)
map_hexbin
```

    User Guide: https://github.com/keplergl/kepler.gl/blob/master/docs/keplergl-jupyter/user-guide.md



    KeplerGl(config={'version': 'v1', 'config': {'visState': {'filters': [{'dataId': 'data', 'id': 'x8ikqz08r', 'n…



```python
with open(KEPLER_CONF_HEXBIN, "w") as fp:
    json.dump(map_hexbin.config, fp)
```

Screenshot:

![Hexbin map](/assets/blog/2020-05-01-evaluation-keplergl-covid-19-data/img/map_hexbin.png "Hexbin map")


```python
HTML('<video controls loop><source src="/assets/blog/2020-05-01-evaluation-keplergl-covid-19-data/vid/vid_hexbin_v1.mp4"></video>')
```




<video controls loop><source src="/assets/blog/2020-05-01-evaluation-keplergl-covid-19-data/vid/vid_hexbin_v1.mp4"></video>



The normalization trick isn't perfect in this setting: as you can see in the legend in the video, the color boundaries change over time. At some point in the timeline, my dummy triangle in the North Sea is no longer the maximum. A hexbin is a sum over multiple neighbourhoods and may be larger than the max of a single neighbourhood. 

A fix isn't trivial, because the binning changes slightly as the map is moved around during playback. In other words: the binning doesn't seem to be deterministic: it is influenced by the view settings.
This is not great from a from a Data Science purism perspective, but it doesn't seem to have much impact on this particular visualization.

# Conclusion

Kepler.gl looks fantastic and can be a great tool for exploring geographic time series interactively. It is especially well suited for datasets of unaggregated events at random intervals. Special care must be taken when using pre-aggregated data on a fixed interval.

The most important things on my wishlist are:
- Ability to aggregate data (mean/sum) if it's on the exact same coordinate (at the very least for Points and Polygons).
- Better support for fixed interval time series (e.g. daily). Getting the graph in the bottom to show something useful would be nice, but most importantly: having the ability to  move the selected time range around when it is relatively small would be very useful.
- A way to ship geo data separately and join it inside Kepler, in order to support large Polygon-based time series where the Polygons themselves remain static over time.
- Built-in support for normalizing across the timeline for the Hexbin (or at least make the binning deterministic such that I can empirically determine the max myself for the      dummy region).

I'll try to reach out to the team to see where they stand on this. For now, I'm happy to start using it in my projects. Thanks for reading!

[Click here for the code](https://github.com/jvanlier/blog-notebooks/tree/master/evaluation-keplergl-covid-19-data).
