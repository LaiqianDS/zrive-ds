import requests
import pandas as pd
import time
import plotly.graph_objects as go
from typing import Optional
from plotly.subplots import make_subplots

API_URL = "https://archive-api.open-meteo.com/v1/archive?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]


def get_api_with_retries(
    url: str, cool_off_time: int = 1, attempts_left: int = 3
) -> Optional[requests.Response]:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response
        if response.status_code == 429:  # Rate limit error
            if attempts_left == 0:
                return None
            time.sleep(cool_off_time)

            # Expontential growth of cool_off_time, arbitrary choice
            new_cool_off_time = cool_off_time * 2
            return get_api_with_retries(
                url, new_cool_off_time, attempts_left=attempts_left - 1
            )

        response.raise_for_status()

    # Handle exceptions
    except requests.exceptions.Timeout as timeout_error:
        print(f"Timeout error occurred: {timeout_error}")
    except requests.exceptions.HTTPError as http_error:  # 4XX, 5XX errors
        print(f"HTTP error occurred: {http_error}")
    except requests.exceptions.RequestException as request_error:  # other errors
        print(f"Request error occurred: {request_error}")
    return None


def get_data_meteo_api(
    city: str, start_date: str = "2010-01-01", end_date: str = "2019-12-31"
) -> Optional[pd.DataFrame]:
    latitude: float = COORDINATES[city]["latitude"]
    longitude: float = COORDINATES[city]["longitude"]
    url: str = (
        f"{API_URL}latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&daily={','.join(VARIABLES)}"
    )
    api_result = get_api_with_retries(url)
    if api_result is None:
        return None
    else:
        data = pd.DataFrame(api_result.json()["daily"])
        return data


def compute_monthly_mean(city_data: pd.DataFrame) -> pd.DataFrame:
    city_data["time"] = pd.to_datetime(city_data["time"])
    city_data["month"] = city_data["time"].dt.month
    city_data["year"] = city_data["time"].dt.year
    city_data["year-month"] = (
        city_data["time"].dt.year.astype(str)
        + "-"
        + city_data["time"].dt.month.astype(str)
    )
    city_data["year-month"] = pd.to_datetime(city_data["year-month"], format="%Y-%m")
    city_data.drop("time", axis=1, inplace=True)
    city_data = city_data.groupby(["year", "month"]).mean().reset_index()

    return city_data


def line_and_bar_plot(df: pd.DataFrame, city: str):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Temperature
    fig.add_trace(
        go.Scatter(
            x=df["year-month"],
            y=df["temperature_2m_mean"],
            mode="lines",
            name="Mean Temperature (°C)",
            line=dict(color="red"),
        ),
        secondary_y=False,
    )

    # Wind
    fig.add_trace(
        go.Scatter(
            x=df["year-month"],
            y=df["wind_speed_10m_max"],
            mode="lines",
            name="Max Wind Speed (m/s)",
            line=dict(color="green"),
        ),
        secondary_y=False,
    )

    # Precipitation
    fig.add_trace(
        go.Bar(
            x=df["year-month"],
            y=df["precipitation_sum"],
            name="Precipitation Sum (mm)",
            opacity=0.6,
            marker=dict(color="blue"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title_text=f"{city} Monthly Mean Temperature, Max Wind Speed, and Precipitation",
        xaxis_title="Year-Month",
    )

    # Left Y Axis
    fig.update_yaxes(
        title_text="Mean Temperature (°C) / Max Wind Speed (m/s)", secondary_y=False
    )

    # Right Y Axis
    fig.update_yaxes(title_text="Precipitation Sum (mm)", secondary_y=True)
    fig.show()


def main():
    madrid_df = get_data_meteo_api("Madrid")
    london_df = get_data_meteo_api("London")
    rio_df = get_data_meteo_api("Rio")
    assert all(
        df is not None for df in [madrid_df, london_df, rio_df]
    ), "An error occurred while fetching data from the API."
    print("Data fetched successfully.")
    madrid_df = compute_monthly_mean(madrid_df)
    london_df = compute_monthly_mean(london_df)
    rio_df = compute_monthly_mean(rio_df)
    line_and_bar_plot(madrid_df, "Madrid")
    line_and_bar_plot(london_df, "London")
    line_and_bar_plot(rio_df, "Rio")


if __name__ == "__main__":
    main()
