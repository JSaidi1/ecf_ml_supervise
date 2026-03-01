import logging
import os
from pathlib import Path
import sys
import inspect
from tabulate import tabulate

from pandas import DataFrame
from pyspark.sql.dataframe import DataFrame

LOG_DIR_PATH = "logs" # default
LOG_FILE_NAME = "app.log" # default

def log_message(
    msg_log: str,
    level: str = "info",
    file_log: bool = False,
    file_log_dir: str = LOG_DIR_PATH,
    file_log_name: str = LOG_FILE_NAME,
    file_log_clear: bool = False,
    console_log: bool = True,
    debug_mode: bool = False,
    set_formatter: bool = True
) -> None:
    """
    Log a message with dynamic configuration per call.

    This function logs a message at the specified severity level and
    dynamically configures logging handlers on each invocation. Depending
    on the provided flags, the message can be sent to the console, a log
    file, or both.

    When `debug_mode` is enabled, the logger and handlers are set to DEBUG
    level and the caller's file name and line number are appended to the
    log message.

    Parameters:
        `level` : Log severity level ("debug", "info", "warning", "error", "critical"). By default = "info"
        `msg_log` (str): The message to be logged.
        `file_log` (bool): If True, log the message to a file located at LOG_DIR_PATH / LOG_FILE_NAME.
        `console_log` (bool): If True, log the message to the console.
        `debug_mode` (bool): If True, enable DEBUG level logging and append caller file and line information to the message.

    Returns:
        None
    """
    if file_log or console_log:
        level = level.lower()
        logger = logging.getLogger("app_logger")
        # --- Update logger level
        logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        
        # === REMOVE existing handlers (important!)
        logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(message)s"
        )

        # === Console log
        if console_log:
            # (optionnel) forcer UTF-8 dans la console
            sys.stdout.reconfigure(encoding="utf-8")

            console_handler = logging.StreamHandler()
            if set_formatter:
                console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # === File log
        if file_log:
            if file_log_name != None and file_log_name != "":
                LOG_FILE_NAME = file_log_name
            Path(file_log_dir).mkdir(parents=True, exist_ok=True)

            # Define log_file_path
            log_file_path = os.path.join(file_log_dir, LOG_FILE_NAME)
            #print("log_file_path =", log_file_path)

            # Clear log file if file_log_clear is True (& log_file_path exists)
            if file_log_clear:
                if os.path.exists(log_file_path):
                    # clear
                    with open(log_file_path, "w"):
                        pass
            #file_handler = logging.FileHandler(os.path.join(file_log_dir, LOG_FILE_NAME))
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            if set_formatter:
                file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # --- Update handler levels every call
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

        # --- Add caller info manually if debug_mode
        if debug_mode:
            frame = inspect.stack()[1]  # caller of log_message
            caller_file = os.path.basename(frame.filename)
            caller_line = frame.lineno
            msg_log = f"{msg_log} - {caller_file}:{caller_line}"

        # --- Dispatch log
        if level == "debug":
            logger.debug(msg_log)
        elif level == "info":
            logger.info(msg_log)
        elif level == "warning":
            logger.warning(msg_log)
        elif level == "error":
            logger.error(msg_log)
        elif level == "critical":
            logger.critical(msg_log)
        else:
            logger.warning(f"Invalid log level '{level}', defaulting to INFO.")
            logger.info(msg_log)
    #else: # => no logging: warning

def log_df(
            msg_log:str,                         # log module  
            df:DataFrame,                        # pandas 
			level:str="info",                    # log module				
			file_log: bool = False,              # log module
            file_log_dir: str = LOG_DIR_PATH,    # log module
			file_log_name: str = LOG_FILE_NAME,  # log module
			file_log_clear: bool = False,        # log module
			console_log: bool = True,            # log module
			debug_mode: bool = False,            # log module
			limit_lines:int=10,                  # pandas
			showindex:bool=True,                 # pandas
			headers:str="keys",                  # pandas
			tablefmt:str="grid",                 # pandas
) -> None:
    """
    Log a formatted preview of a DataFrame using the configured logging system.

    This function converts the first rows of a DataFrame into a human-readable
    table (via `tabulate`) and appends it to a log message. The output can be
    written to the console, a log file, or both, depending on the logging options.

    Parameters
    ----------
    msg_log : str
        Base log message displayed before the DataFrame preview.
    df : DataFrame
        DataFrame to be logged (Spark or pandas-compatible via `toPandas()`).
    level : str, optional
        Logging level (e.g. "info", "warning", "error", "debug"), by default "info".
    file_log : bool, optional
        Whether to write the log message to a file, by default False.
    file_log_name : str, optional
        Log file name or path, by default LOG_FILE_NAME.
    file_log_clear : bool, optional
        Whether to clear the log file before writing, by default False.
    console_log : bool, optional
        Whether to output the log message to the console, by default True.
    debug_mode : bool, optional
        Enable debug-specific logging behavior, by default False.
    limit_lines : int, optional
        Maximum number of DataFrame rows to include in the log, by default 10.
    showindex : bool, optional
        Whether to display the DataFrame index in the table, by default True.
    headers : str, optional
        Header configuration passed to `tabulate`, by default "keys".
    tablefmt : str, optional
        Table format passed to `tabulate` (e.g. "grid", "psql", "plain"),
        by default "grid".

    Returns
    -------
    None
        This function does not return anything; it only writes logs.
    """
	
	# define tabulate config (pandas requirements)
    sample_txt = tabulate(
        df.limit(limit_lines).toPandas(),
        headers=headers,
        tablefmt=tablefmt,
        showindex=showindex
    )
	
    msg_log_df = msg_log + f"\n{sample_txt}"
	
    # write log
    log_message(level=level, 
	            msg_log=msg_log_df, 
	            file_log=file_log, 
                file_log_dir=file_log_dir,
				file_log_name=file_log_name, 
				file_log_clear=file_log_clear, 
				console_log=console_log, 
				debug_mode=debug_mode,
				)
	

def log_table(
        file_log: str,       # log fnc
        file_log_dir: str,   # log fnc
		file_log_name: str,  # log fnc
        data: dict,          # current fnc
        headers=None         # current fnc
        ):
    """
    Create table in logs. 

    data: list[dict]  -> [{"col1": "...", "col2": "..."}]

    headers: list[str] (optionnel)
    """
    if not data:
        return

    if headers is None:
        headers = list(data[0].keys())

    # --- NEW: headers multi-lignes ---
    header_lines = {h: str(h).split("\n") for h in headers}
    header_height = max(len(lines) for lines in header_lines.values())

    # Convertir chaque cellule en liste de lignes (split \n)
    processed = []
    for row in data:
        processed.append({h: str(row.get(h, "")).split("\n") for h in headers})

    # Largeur auto (basée sur la plus longue ligne) + titres
    widths = []
    for h in headers:
        max_cell = max(len(line) for row in processed for line in row[h]) if processed else 0
        max_head = max(len(line) for line in header_lines[h])
        widths.append(max(max_cell, max_head) + 2)

    def sep():
        return "+" + "+".join("-" * w for w in widths) + "+"

    # --- imprime le header sur plusieurs lignes ---
    def log_header():
        for i in range(header_height):
            line = "|"
            for h, w in zip(headers, widths):
                lines = header_lines[h]
                text = lines[i] if i < len(lines) else ""
                line += f"{text:^{w}}|"
            log_message(level="info", msg_log=line, file_log=file_log, file_log_dir=file_log_dir, file_log_name=file_log_name)

    log_message(level="info", msg_log=sep(), file_log=file_log, file_log_dir=file_log_dir, file_log_name=file_log_name)
    log_header()
    log_message(level="info", msg_log=sep(), file_log=file_log, file_log_dir=file_log_dir, file_log_name=file_log_name)

    # Affichage lignes (données)
    for row in processed:
        height = max(len(row[h]) for h in headers)

        for i in range(height):
            line = "|"
            for h, w in zip(headers, widths):
                cell_lines = row[h]
                text = cell_lines[i] if i < len(cell_lines) else ""
                line += f"{text:<{w}}|"
            log_message(level="info", msg_log=line, file_log=file_log, file_log_dir=file_log_dir, file_log_name=file_log_name)

        log_message(level="info", msg_log=sep(), file_log=file_log, file_log_dir=file_log_dir, file_log_name=file_log_name)

