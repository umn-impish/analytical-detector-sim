"""Magic data so we can programmatically request information from NIST XCOM."""

import random
import string

# Encode this directly with newlines and carriage returns so that the NIST HTTP
# server doesn't crap itself
boundary = "pleaseohgodsplitmeup"


def generate_request(znum: int) -> dict[str, object]:
    """Generate a NIST-compatible information request with a random boundary separator"""
    # Mozilla-style boundary
    boundary = ("-" * 27) + "".join(random.choices(population=string.digits, k=28))
    request = f"""--{boundary}\r\nContent-Disposition: form-data; name="character"\r\n\r\ntab\r\n--{boundary}\r\nContent-Disposition: form-data; name="Name"\r\n\r\nCarbon\r\n--{boundary}\r\nContent-Disposition: form-data; name="Method"\r\n\r\n1\r\n--{boundary}\r\nContent-Disposition: form-data; name="ZNum"\r\n\r\n{znum}\r\n--{boundary}\r\nContent-Disposition: form-data; name="OutOpt"\r\n\r\nPIC\r\n--{boundary}\r\nContent-Disposition: form-data; name="NumAdd"\r\n\r\n1\r\n--{boundary}\r\nContent-Disposition: form-data; name="Energies"\r\n\r\n\r\n--{boundary}\r\nContent-Disposition: form-data; name="Output"\r\n\r\non\r\n--{boundary}\r\nContent-Disposition: form-data; name="WindowXmin"\r\n\r\n0.001\r\n--{boundary}\r\nContent-Disposition: form-data; name="WindowXmax"\r\n\r\n100000\r\n--{boundary}\r\nContent-Disposition: form-data; name="photoelectric"\r\n\r\non\r\n--{boundary}\r\nContent-Disposition: form-data; name="coherent"\r\n\r\non\r\n--{boundary}\r\nContent-Disposition: form-data; name="incoherent"\r\n\r\non\r\n--{boundary}--\r\n"""

    # Need to specify these for NIST to be happy
    # (we are basically spoofing the browser)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Origin": "https://physics.nist.gov",
        "Connection": "keep-alive",
        "Referer": "https://physics.nist.gov/cgi-bin/Xcom/xcom3_1",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Priority": "u=0, i",
    }

    return {"request": request, "headers": headers}
