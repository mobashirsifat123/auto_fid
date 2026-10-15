"""Export the API schema without starting a server."""

import json

from .app import create_app


def main() -> None:
    print(json.dumps(create_app().openapi(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
