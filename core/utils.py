import logging

from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        RichHandler(
            console=Console(force_terminal=True),
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
        )
    ],
)

openalex_api_url = "https://api.openalex.org/works"
scopus_api_key = "<API KEY>"
umls_api_key = "<API KEY>"
user_email = "<EMAIL>"
