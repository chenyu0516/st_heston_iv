from datetime import datetime, timezone

def time_format_transform(time:str, timezone=timezone.utc):
    """
    change the string to like "2023-10-27T08:00:00.000Z"
    to the standard datetime
    """
    dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ")
    # By default, this is a naive datetime. If you want it to be aware (UTC), add tzinfo:
    return dt.replace(tzinfo=timezone.utc)

def time_type_change(time:str):
    """
    change the string to like "2023-09-16 02:00:00+00:00"
    to datetime
    """
    return datetime.strptime(time, "%Y-%m-%d %H:%M:%S%z")
