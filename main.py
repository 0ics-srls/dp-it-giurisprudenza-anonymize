"""Entrypoint of dp-it-giurisprudenza-anonymize: proc-protocol stdin loop. Action ``execute`` anonymizes the plan files of the given items;
``terminate`` (or stdin EOF) shuts down. Data-only: reads and writes the data directory, no database. stdout carries protocol events only."""
from __future__ import annotations

import logging
import sys
from typing import TextIO

import procclass
from dp_it_giurisprudenza_anonymize.dispatch import CONFIG_ERROR, SYSTEM_ERROR, AnonDeps, AnonError, run_anonymize

log = logging.getLogger("dp_it_giurisprudenza_anonymize")


def main(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout, deps: AnonDeps | None = None) -> int:
    """One JSON request per stdin line until terminate/EOF; returns 1 if any request failed."""
    ports = deps or AnonDeps()
    exit_code = 0
    for line in stdin:
        if not line.strip():
            continue
        try:
            request = procclass.parse(line)
        except procclass.ProcProtocolError as ex:
            procclass.emit_error(stdout, CONFIG_ERROR, str(ex))
            exit_code = 1
            continue
        if request.action == "terminate":
            break
        if request.action != "execute":
            procclass.emit_error(stdout, CONFIG_ERROR, f"unknown action: {request.action}")
            exit_code = 1
            continue
        try:
            result = run_anonymize(request.params, ports, lambda percent, message: procclass.emit_progress(stdout, percent, message))
        except AnonError as error:
            procclass.emit_error(stdout, error.code, str(error))
            exit_code = 1
            continue
        except Exception:
            log.exception("unexpected failure in execute")
            procclass.emit_error(stdout, SYSTEM_ERROR, "unexpected failure in execute")
            exit_code = 1
            continue
        procclass.emit_result(stdout, result)
    return exit_code


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
