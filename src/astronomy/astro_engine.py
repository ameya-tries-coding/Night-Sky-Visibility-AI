from skyfield.api import load, Topos

ts = load.timescale()
planets = load('de421.bsp')

def get_real_objects(lat, lon, selected_date, hour):
    t = ts.utc(selected_date.year, selected_date.month, selected_date.day, hour)

    observer = Topos(latitude_degrees=lat, longitude_degrees=lon)
    earth = planets['earth']

    objects = {
        "Mars": planets['mars'],
        "Jupiter": planets['jupiter barycenter'],
        "Venus": planets['venus'],
        "Saturn": planets['saturn barycenter'],
        "Moon": planets['moon']
    }

    data = []

    for name, obj in objects.items():
        astrometric = (earth + observer).at(t).observe(obj)
        alt, az, _ = astrometric.apparent().altaz()

        if alt.degrees > -5:
            data.append({
                "event_type": "planet",
                "constellation": name,
                "elevation_deg": alt.degrees,
                "azimuth_deg": az.degrees,
                "estimated_magnitude": 1.5
            })

    return data