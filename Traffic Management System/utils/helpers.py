# utils/helpers.py
"""
General utility functions for the Smart Traffic Management System.

This module provides common helper functions used throughout the application.
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# DICTIONARY UTILITIES
# =============================================================================

def safe_get(
    d: Any,
    key: str,
    default: T = None
) -> Union[Any, T]:
    """
    Safely get a value from a dictionary.
    
    Args:
        d: Dictionary (or any object)
        key: Key to retrieve
        default: Default value if key not found or d is not a dict
        
    Returns:
        Value for key or default
        
    Examples:
        >>> safe_get({'a': 1}, 'a')
        1
        >>> safe_get({'a': 1}, 'b', 'default')
        'default'
        >>> safe_get(None, 'a', 'default')
        'default'
    """
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def deep_get(
    d: Dict,
    keys: Union[str, List[str]],
    default: T = None,
    separator: str = '.'
) -> Union[Any, T]:
    """
    Safely get a nested value from a dictionary.
    
    Args:
        d: Dictionary to search
        keys: Dot-separated string or list of keys
        default: Default value if path not found
        separator: Key separator (default: '.')
        
    Returns:
        Nested value or default
        
    Examples:
        >>> data = {'a': {'b': {'c': 1}}}
        >>> deep_get(data, 'a.b.c')
        1
        >>> deep_get(data, ['a', 'b', 'c'])
        1
        >>> deep_get(data, 'a.b.x', 'not found')
        'not found'
    """
    if not isinstance(d, dict):
        return default
    
    if isinstance(keys, str):
        keys = keys.split(separator)
    
    result = d
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
        elif isinstance(result, (list, tuple)) and key.isdigit():
            try:
                result = result[int(key)]
            except (IndexError, TypeError):
                return default
        else:
            return default
        
        if result is None:
            return default
    
    return result


def deep_set(
    d: Dict,
    keys: Union[str, List[str]],
    value: Any,
    separator: str = '.'
) -> Dict:
    """
    Set a nested value in a dictionary, creating intermediate dicts as needed.
    
    Args:
        d: Dictionary to modify
        keys: Dot-separated string or list of keys
        value: Value to set
        separator: Key separator
        
    Returns:
        Modified dictionary
        
    Examples:
        >>> data = {}
        >>> deep_set(data, 'a.b.c', 1)
        {'a': {'b': {'c': 1}}}
    """
    if isinstance(keys, str):
        keys = keys.split(separator)
    
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value
    return d


def merge_dicts(
    *dicts: Dict,
    deep: bool = True
) -> Dict:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge
        deep: Whether to merge nested dicts recursively
        
    Returns:
        Merged dictionary
        
    Examples:
        >>> merge_dicts({'a': 1}, {'b': 2})
        {'a': 1, 'b': 2}
        >>> merge_dicts({'a': {'x': 1}}, {'a': {'y': 2}}, deep=True)
        {'a': {'x': 1, 'y': 2}}
    """
    result = {}
    
    for d in dicts:
        if not isinstance(d, dict):
            continue
        
        for key, value in d.items():
            if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value, deep=True)
            else:
                result[key] = value
    
    return result


def flatten_dict(
    d: Dict,
    parent_key: str = '',
    separator: str = '.'
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys (used in recursion)
        separator: Separator for nested keys
        
    Returns:
        Flattened dictionary
        
    Examples:
        >>> flatten_dict({'a': {'b': 1, 'c': 2}})
        {'a.b': 1, 'a.c': 2}
    """
    items = []
    for key, value in d.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


# =============================================================================
# STRING UTILITIES
# =============================================================================

def sanitize_string(
    s: str,
    allowed_chars: str = r'a-zA-Z0-9\s\-_',
    replacement: str = '',
    max_length: Optional[int] = None
) -> str:
    """
    Sanitize a string by removing/replacing unwanted characters.
    
    Args:
        s: String to sanitize
        allowed_chars: Regex pattern for allowed characters
        replacement: Replacement for disallowed characters
        max_length: Maximum length (truncate if exceeded)
        
    Returns:
        Sanitized string
        
    Examples:
        >>> sanitize_string("Hello, World!")
        'Hello World'
        >>> sanitize_string("<script>alert('xss')</script>")
        "scriptalert'xss'script"
    """
    if not isinstance(s, str):
        return ''
    
    # Remove disallowed characters
    pattern = f'[^{allowed_chars}]'
    result = re.sub(pattern, replacement, s)
    
    # Collapse multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    # Truncate if needed
    if max_length and len(result) > max_length:
        result = result[:max_length].rsplit(' ', 1)[0]
    
    return result


def truncate(
    s: str,
    max_length: int = 100,
    suffix: str = '...'
) -> str:
    """
    Truncate a string to a maximum length.
    
    Args:
        s: String to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated string
        
    Examples:
        >>> truncate("Hello World", 8)
        'Hello...'
    """
    if not isinstance(s, str):
        return ''
    
    if len(s) <= max_length:
        return s
    
    return s[:max_length - len(suffix)] + suffix


def slugify(s: str, separator: str = '-') -> str:
    """
    Convert a string to a URL-friendly slug.
    
    Args:
        s: String to slugify
        separator: Word separator
        
    Returns:
        Slugified string
        
    Examples:
        >>> slugify("Hello World!")
        'hello-world'
        >>> slugify("Traffic @ 10:00 AM")
        'traffic-10-00-am'
    """
    if not isinstance(s, str):
        return ''
    
    # Convert to lowercase
    s = s.lower()
    
    # Replace spaces and special chars with separator
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s_]+', separator, s)
    
    # Remove leading/trailing separators
    s = s.strip(separator)
    
    return s


def mask_sensitive(
    s: str,
    visible_chars: int = 4,
    mask_char: str = '*'
) -> str:
    """
    Mask a sensitive string (like API keys).
    
    Args:
        s: String to mask
        visible_chars: Number of visible characters at end
        mask_char: Character to use for masking
        
    Returns:
        Masked string
        
    Examples:
        >>> mask_sensitive("sk-abc123xyz789")
        '***********789'
    """
    if not isinstance(s, str) or len(s) <= visible_chars:
        return mask_char * len(s) if s else ''
    
    masked_length = len(s) - visible_chars
    return mask_char * masked_length + s[-visible_chars:]


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def is_valid_email(email: str) -> bool:
    """Check if string is a valid email format."""
    if not isinstance(email, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """Check if string is a valid URL format."""
    if not isinstance(url, str):
        return False
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url, re.IGNORECASE))


def is_valid_coordinates(lat: float, lon: float) -> bool:
    """Check if coordinates are valid."""
    try:
        return -90 <= float(lat) <= 90 and -180 <= float(lon) <= 180
    except (TypeError, ValueError):
        return False


def is_valid_api_key(key: str, min_length: int = 20) -> bool:
    """Check if string looks like a valid API key."""
    if not isinstance(key, str):
        return False
    return len(key) >= min_length and key.replace('-', '').replace('_', '').isalnum()


# =============================================================================
# TIME UTILITIES
# =============================================================================

def now_iso() -> str:
    """Get current time in ISO format."""
    return datetime.now().isoformat()


def now_utc_iso() -> str:
    """Get current UTC time in ISO format."""
    return datetime.utcnow().isoformat() + 'Z'


def parse_datetime(
    dt_string: str,
    formats: Optional[List[str]] = None
) -> Optional[datetime]:
    """
    Parse a datetime string with multiple format attempts.
    
    Args:
        dt_string: Datetime string to parse
        formats: List of formats to try
        
    Returns:
        Parsed datetime or None
    """
    if not dt_string:
        return None
    
    formats = formats or [
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(dt_string, fmt)
        except ValueError:
            continue
    
    return None


def format_duration(seconds: Union[int, float]) -> str:
    """
    Format seconds into human-readable duration.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "2h 30m" or "45s"
        
    Examples:
        >>> format_duration(3661)
        '1h 1m 1s'
        >>> format_duration(45)
        '45s'
    """
    if seconds < 0:
        return '0s'
    
    seconds = int(seconds)
    
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    
    parts = []
    if hours:
        parts.append(f'{hours}h')
    if minutes:
        parts.append(f'{minutes}m')
    if secs or not parts:
        parts.append(f'{secs}s')
    
    return ' '.join(parts)


def time_ago(dt: datetime) -> str:
    """
    Get human-readable time difference from now.
    
    Args:
        dt: Datetime to compare
        
    Returns:
        String like "5 minutes ago" or "2 days ago"
    """
    now = datetime.now()
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return 'just now'
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f'{minutes} minute{"s" if minutes != 1 else ""} ago'
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f'{hours} hour{"s" if hours != 1 else ""} ago'
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f'{days} day{"s" if days != 1 else ""} ago'
    else:
        weeks = int(seconds / 604800)
        return f'{weeks} week{"s" if weeks != 1 else ""} ago'


# =============================================================================
# DECORATORS
# =============================================================================

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,)
) -> Callable:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts
        backoff: Multiplier for delay after each attempt
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorated function
        
    Examples:
        @retry(max_attempts=3, delay=1.0)
        def unstable_api_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    return decorator


def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Examples:
        @timer
        def slow_function():
            ...
        # Logs: "slow_function took 1.234s"
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def deprecated(message: str = '') -> Callable:
    """
    Mark a function as deprecated.
    
    Args:
        message: Deprecation message
        
    Examples:
        @deprecated("Use new_function instead")
        def old_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import warnings
            warnings.warn(
                f"{func.__name__} is deprecated. {message}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def memoize(func: Callable) -> Callable:
    """
    Simple memoization decorator for functions with hashable arguments.
    
    Examples:
        @memoize
        def expensive_calculation(x, y):
            ...
    """
    cache = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    
    wrapper.cache = cache
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper


# =============================================================================
# FILE UTILITIES
# =============================================================================

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(
    file_path: Union[str, Path],
    algorithm: str = 'md5'
) -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256', etc.)
        
    Returns:
        Hex digest of file hash
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def read_json(
    file_path: Union[str, Path],
    default: T = None
) -> Union[Dict, List, T]:
    """
    Read JSON file safely.
    
    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Parsed JSON or default
    """
    path = Path(file_path)
    
    if not path.exists():
        return default
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error reading JSON from {path}: {e}")
        return default


def write_json(
    data: Union[Dict, List],
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False
) -> bool:
    """
    Write data to JSON file.
    
    Args:
        data: Data to write
        file_path: Path to output file
        indent: JSON indentation
        ensure_ascii: Whether to escape non-ASCII characters
        
    Returns:
        True if successful
    """
    path = Path(file_path)
    
    try:
        ensure_dir(path.parent)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        return True
    except IOError as e:
        logger.error(f"Error writing JSON to {path}: {e}")
        return False


# =============================================================================
# COLLECTION UTILITIES
# =============================================================================

def chunk_list(
    lst: List[T],
    chunk_size: int
) -> Iterable[List[T]]:
    """
    Split a list into chunks.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Yields:
        List chunks
        
    Examples:
        >>> list(chunk_list([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def unique_ordered(lst: List[T]) -> List[T]:
    """
    Remove duplicates while preserving order.
    
    Args:
        lst: List with possible duplicates
        
    Returns:
        List with duplicates removed
        
    Examples:
        >>> unique_ordered([1, 2, 2, 3, 1, 4])
        [1, 2, 3, 4]
    """
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]


def first(
    iterable: Iterable[T],
    default: T = None,
    predicate: Callable[[T], bool] = None
) -> T:
    """
    Get first element from iterable, optionally matching a predicate.
    
    Args:
        iterable: Iterable to search
        default: Default if not found
        predicate: Optional filter function
        
    Returns:
        First matching element or default
        
    Examples:
        >>> first([1, 2, 3])
        1
        >>> first([1, 2, 3], predicate=lambda x: x > 1)
        2
    """
    if predicate:
        return next((x for x in iterable if predicate(x)), default)
    return next(iter(iterable), default)


# =============================================================================
# ENVIRONMENT UTILITIES
# =============================================================================

def get_env(
    key: str,
    default: T = None,
    required: bool = False,
    cast: Callable = None
) -> Union[str, T]:
    """
    Get environment variable with optional type casting.
    
    Args:
        key: Environment variable name
        default: Default value
        required: Raise error if missing
        cast: Function to cast value (e.g., int, float, bool)
        
    Returns:
        Environment variable value
        
    Raises:
        ValueError: If required and missing
        
    Examples:
        >>> get_env('PORT', 8080, cast=int)
        8080
        >>> get_env('DEBUG', False, cast=lambda x: x.lower() == 'true')
        False
    """
    value = os.getenv(key)
    
    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return default
    
    if cast:
        try:
            return cast(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to cast {key}={value}: {e}")
            return default
    
    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, '').lower()
    if value in ('true', '1', 'yes', 'on'):
        return True
    if value in ('false', '0', 'no', 'off'):
        return False
    return default


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    return get_env(key, default, cast=int)


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable."""
    return get_env(key, default, cast=float)


def get_env_list(
    key: str,
    default: List[str] = None,
    separator: str = ','
) -> List[str]:
    """
    Get list from environment variable.
    
    Args:
        key: Environment variable name
        default: Default list
        separator: Value separator
        
    Returns:
        List of values
        
    Examples:
        # ALLOWED_HOSTS=host1.com,host2.com
        >>> get_env_list('ALLOWED_HOSTS')
        ['host1.com', 'host2.com']
    """
    value = os.getenv(key)
    if value is None:
        return default or []
    return [v.strip() for v in value.split(separator) if v.strip()]


# =============================================================================
# HASHING & ENCODING
# =============================================================================

def hash_string(
    s: str,
    algorithm: str = 'md5'
) -> str:
    """
    Hash a string.
    
    Args:
        s: String to hash
        algorithm: Hash algorithm
        
    Returns:
        Hex digest
    """
    return hashlib.new(algorithm, s.encode()).hexdigest()


def generate_id(
    length: int = 8,
    prefix: str = ''
) -> str:
    """
    Generate a random ID.
    
    Args:
        length: Length of random part
        prefix: Optional prefix
        
    Returns:
        Generated ID
    """
    import secrets
    random_part = secrets.token_hex(length // 2)
    return f"{prefix}{random_part}" if prefix else random_part


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logging(
    level: str = 'INFO',
    format_string: str = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Log level
        format_string: Custom format string
        log_file: Optional file to log to
        
    Returns:
        Root logger
    """
    format_string = format_string or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        ensure_dir(Path(log_file).parent)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers
    )
    
    return logging.getLogger()


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    # Dictionary
    'safe_get',
    'deep_get',
    'deep_set',
    'merge_dicts',
    'flatten_dict',
    # String
    'sanitize_string',
    'truncate',
    'slugify',
    'mask_sensitive',
    # Validation
    'is_valid_email',
    'is_valid_url',
    'is_valid_coordinates',
    'is_valid_api_key',
    # Time
    'now_iso',
    'now_utc_iso',
    'parse_datetime',
    'format_duration',
    'time_ago',
    # Decorators
    'retry',
    'timer',
    'deprecated',
    'memoize',
    # File
    'ensure_dir',
    'get_file_hash',
    'read_json',
    'write_json',
    # Collection
    'chunk_list',
    'unique_ordered',
    'first',
    # Environment
    'get_env',
    'get_env_bool',
    'get_env_int',
    'get_env_float',
    'get_env_list',
    # Hashing
    'hash_string',
    'generate_id',
    # Logging
    'setup_logging',
]
