def get_direction(azimuth):
    if azimuth >= 337.5 or azimuth < 22.5:
        return "North"
    elif azimuth < 67.5:
        return "North-East"
    elif azimuth < 112.5:
        return "East"
    elif azimuth < 157.5:
        return "South-East"
    elif azimuth < 202.5:
        return "South"
    elif azimuth < 247.5:
        return "South-West"
    elif azimuth < 292.5:
        return "West"
    else:
        return "North-West"


def get_altitude_label(elevation):
    if elevation > 70:
        return "Near Zenith (Above Head)"
    elif elevation > 40:
        return "Mid Sky"
    else:
        return "Near Horizon"


def rank_objects(df, top_n=5):
    df = df.copy()

    # Add direction + altitude info
    df["direction"] = df["azimuth_deg"].apply(get_direction)
    df["sky_position"] = df["elevation_deg"].apply(get_altitude_label)

    ranked = df.sort_values(by="visibility_score", ascending=False)

    return ranked.head(top_n)