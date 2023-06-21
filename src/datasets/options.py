DATE_ISO_REGEX = r"^\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\dZ"
MODEL_URI_REGEX = (
    r"^s3:\/\/([^/\s]+)\/([\w\W]+)\.(.*)|^models:\/[^/\s]+\/(\d+|Production|Staging)|^runs:\/[^/\s]+\/model"
)
ALPHANUMERIC_REGEX = r"^[A-Za-z0-9]+"
