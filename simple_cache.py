"""
simple_cache.py - Enhanced Caching System with Multi-Type Support
OPTIMIZED: Supports multiple cache types with variable TTLs, cache statistics, and namespace isolation
"""

import hashlib
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict
from threading import Lock
from datetime import datetime, timedelta

# Import cache configuration from constants
try:
    from constants import (
        CACHE_TTL_BY_TYPE,
        CACHE_MAX_SIZE_BY_TYPE,
        CACHE_KEY_PREFIXES,
        FEATURES
    )
except ImportError:
    # Fallback defaults if constants not available
    CACHE_TTL_BY_TYPE = {'default': 3600}
    CACHE_MAX_SIZE_BY_TYPE = {'default': 100}
    CACHE_KEY_PREFIXES = {'default': ''}
    FEATURES = {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# CACHE ENTRY CLASS
# ============================================

class CacheEntry:
    """Individual cache entry with metadata"""
    
    def __init__(self, key: str, value: Any, ttl: int, cache_type: str = 'default'):
        self.key = key
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl
        self.cache_type = cache_type
        self.hit_count = 0
        self.last_accessed = self.timestamp
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return time.time() - self.timestamp > self.ttl
    
    def access(self) -> Any:
        """Record access and return value"""
        self.hit_count += 1
        self.last_accessed = time.time()
        return self.value
    
    def age(self) -> float:
        """Get age of entry in seconds"""
        return time.time() - self.timestamp
    
    def time_to_live(self) -> float:
        """Get remaining TTL in seconds"""
        remaining = self.ttl - self.age()
        return max(0, remaining)

# ============================================
# ENHANCED QUERY CACHE
# ============================================

class QueryCache:
    """
    Enhanced cache with multi-type support, statistics, and namespace isolation.
    Each cache type can have different TTL and size limits.
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600, 
                 cache_type: str = 'default', enable_stats: bool = True):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time to live in seconds
            cache_type: Type of cache (for namespace isolation)
            enable_stats: Whether to track statistics
        """
        # Use configuration from constants if available
        self.cache_type = cache_type
        self.max_size = CACHE_MAX_SIZE_BY_TYPE.get(cache_type, max_size)
        self.ttl = CACHE_TTL_BY_TYPE.get(cache_type, ttl_seconds)
        
        # Cache storage (using OrderedDict for LRU)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Thread safety
        self.lock = Lock()
        
        # Statistics
        self.enable_stats = enable_stats
        if self.enable_stats:
            self.stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'expirations': 0,
                'total_requests': 0,
                'cache_type': cache_type,
                'created_at': datetime.now()
            }
        
        logger.info(f"QueryCache initialized: type={cache_type}, max_size={self.max_size}, ttl={self.ttl}s")
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for key with optional prefix"""
        # Add prefix based on cache type
        prefix = CACHE_KEY_PREFIXES.get(self.cache_type, '')
        prefixed_key = f"{prefix}{key}"
        
        # Hash for consistent length
        return hashlib.md5(prefixed_key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if exists and not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        hashed_key = self._hash_key(key)
        
        with self.lock:
            if self.enable_stats:
                self.stats['total_requests'] += 1
            
            if hashed_key in self.cache:
                entry = self.cache[hashed_key]
                
                if entry.is_expired():
                    # Remove expired entry
                    del self.cache[hashed_key]
                    if self.enable_stats:
                        self.stats['expirations'] += 1
                        self.stats['misses'] += 1
                    logger.debug(f"Cache miss (expired): {key[:20]}...")
                    return None
                else:
                    # Move to end (LRU)
                    self.cache.move_to_end(hashed_key)
                    
                    # Record hit
                    if self.enable_stats:
                        self.stats['hits'] += 1
                    
                    value = entry.access()
                    logger.debug(f"Cache hit: {key[:20]}... (hits: {entry.hit_count})")
                    return value
            
            # Cache miss
            if self.enable_stats:
                self.stats['misses'] += 1
            logger.debug(f"Cache miss: {key[:20]}...")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache with optional custom TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL (uses default if not provided)
        """
        hashed_key = self._hash_key(key)
        entry_ttl = ttl if ttl is not None else self.ttl
        
        with self.lock:
            # Remove oldest entry if cache is full (LRU eviction)
            if len(self.cache) >= self.max_size and hashed_key not in self.cache:
                # Find oldest entry (first in OrderedDict)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                
                if self.enable_stats:
                    self.stats['evictions'] += 1
                logger.debug(f"Cache eviction (LRU): removing oldest entry")
            
            # Add/update entry
            entry = CacheEntry(key, value, entry_ttl, self.cache_type)
            self.cache[hashed_key] = entry
            
            # Move to end (most recently used)
            self.cache.move_to_end(hashed_key)
            
            logger.debug(f"Cache set: {key[:20]}... (TTL: {entry_ttl}s)")
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        hashed_key = self._hash_key(key)
        
        with self.lock:
            if hashed_key in self.cache:
                del self.cache[hashed_key]
                logger.debug(f"Cache delete: {key[:20]}...")
                return True
            return False
    
    def clear(self) -> int:
        """
        Clear all entries from cache.
        
        Returns:
            Number of entries cleared
        """
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"Cache cleared: {count} entries removed")
            return count
    
    def clean_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                if self.enable_stats:
                    self.stats['expirations'] += 1
            
            if expired_keys:
                logger.info(f"Cleaned {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)
    
    def is_full(self) -> bool:
        """Check if cache is at maximum capacity"""
        return len(self.cache) >= self.max_size
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.enable_stats:
            return {'stats_enabled': False}
        
        with self.lock:
            stats = self.stats.copy()
            
            # Calculate derived metrics
            total = stats['total_requests']
            if total > 0:
                stats['hit_rate'] = stats['hits'] / total
                stats['miss_rate'] = stats['misses'] / total
            else:
                stats['hit_rate'] = 0.0
                stats['miss_rate'] = 0.0
            
            # Add current state
            stats['current_size'] = len(self.cache)
            stats['max_size'] = self.max_size
            stats['utilization'] = len(self.cache) / self.max_size if self.max_size > 0 else 0
            stats['ttl'] = self.ttl
            
            # Add age statistics
            if self.cache:
                ages = [entry.age() for entry in self.cache.values()]
                stats['avg_age'] = sum(ages) / len(ages)
                stats['oldest_age'] = max(ages)
                stats['newest_age'] = min(ages)
            
            return stats
    
    def get_entries_info(self) -> List[Dict[str, Any]]:
        """
        Get information about cached entries.
        
        Returns:
            List of entry information dictionaries
        """
        with self.lock:
            entries = []
            for hashed_key, entry in self.cache.items():
                entries.append({
                    'key': entry.key[:50],  # Truncate for privacy
                    'cache_type': entry.cache_type,
                    'age': entry.age(),
                    'ttl_remaining': entry.time_to_live(),
                    'hit_count': entry.hit_count,
                    'size_estimate': len(str(entry.value))
                })
            
            # Sort by hit count (most used first)
            entries.sort(key=lambda x: x['hit_count'], reverse=True)
            
            return entries
    
    def optimize(self) -> Dict[str, int]:
        """
        Optimize cache by removing expired entries and reordering.
        
        Returns:
            Optimization results
        """
        with self.lock:
            initial_size = len(self.cache)
            
            # Remove expired entries
            expired = self.clean_expired()
            
            # Reorder by access frequency (optional)
            if len(self.cache) > 0:
                # Sort by hit count and last accessed
                sorted_items = sorted(
                    self.cache.items(),
                    key=lambda x: (x[1].hit_count, x[1].last_accessed),
                    reverse=True
                )
                self.cache.clear()
                for key, entry in sorted_items:
                    self.cache[key] = entry
            
            final_size = len(self.cache)
            
            logger.info(f"Cache optimized: {initial_size} → {final_size} entries")
            
            return {
                'expired_removed': expired,
                'initial_size': initial_size,
                'final_size': final_size
            }

# ============================================
# MULTI-CACHE MANAGER
# ============================================

class MultiCacheManager:
    """
    Manager for multiple cache instances with different types.
    Provides centralized access to all cache types.
    """
    
    def __init__(self):
        """Initialize the multi-cache manager"""
        self.caches: Dict[str, QueryCache] = {}
        self.lock = Lock()
        
        # Initialize caches based on configuration
        if FEATURES.get('granular_caching', False):
            self._initialize_configured_caches()
        else:
            # Create single default cache
            self.caches['default'] = QueryCache(
                cache_type='default',
                enable_stats=True
            )
        
        logger.info(f"MultiCacheManager initialized with {len(self.caches)} cache types")
    
    def _initialize_configured_caches(self) -> None:
        """Initialize caches based on configuration from constants"""
        for cache_type in CACHE_TTL_BY_TYPE.keys():
            if cache_type not in self.caches:
                self.caches[cache_type] = QueryCache(
                    max_size=CACHE_MAX_SIZE_BY_TYPE.get(cache_type, 100),
                    ttl_seconds=CACHE_TTL_BY_TYPE.get(cache_type, 3600),
                    cache_type=cache_type,
                    enable_stats=True
                )
                logger.info(f"Initialized cache: {cache_type}")
    
    def get_cache(self, cache_type: str) -> Optional[QueryCache]:
        """
        Get cache instance by type.
        
        Args:
            cache_type: Type of cache
            
        Returns:
            QueryCache instance or None
        """
        return self.caches.get(cache_type, self.caches.get('default'))
    
    def get(self, key: str, cache_type: str = 'default') -> Optional[Any]:
        """Get value from specific cache type"""
        cache = self.get_cache(cache_type)
        if cache:
            return cache.get(key)
        return None
    
    def set(self, key: str, value: Any, cache_type: str = 'default', ttl: Optional[int] = None) -> None:
        """Set value in specific cache type"""
        cache = self.get_cache(cache_type)
        if cache:
            cache.set(key, value, ttl)
    
    def delete(self, key: str, cache_type: str = 'default') -> bool:
        """Delete from specific cache type"""
        cache = self.get_cache(cache_type)
        if cache:
            return cache.delete(key)
        return False
    
    def clear_all(self) -> Dict[str, int]:
        """
        Clear all caches.
        
        Returns:
            Count of entries cleared per cache type
        """
        results = {}
        for cache_type, cache in self.caches.items():
            results[cache_type] = cache.clear()
        
        logger.info(f"All caches cleared: {results}")
        return results
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all caches.
        
        Returns:
            Statistics per cache type
        """
        stats = {}
        for cache_type, cache in self.caches.items():
            stats[cache_type] = cache.get_stats()
        
        # Add aggregate statistics
        total_hits = sum(s.get('hits', 0) for s in stats.values())
        total_misses = sum(s.get('misses', 0) for s in stats.values())
        total_requests = sum(s.get('total_requests', 0) for s in stats.values())
        
        stats['_aggregate'] = {
            'total_hits': total_hits,
            'total_misses': total_misses,
            'total_requests': total_requests,
            'overall_hit_rate': total_hits / total_requests if total_requests > 0 else 0,
            'cache_count': len(self.caches),
            'total_entries': sum(s.get('current_size', 0) for s in stats.values())
        }
        
        return stats
    
    def optimize_all(self) -> Dict[str, Dict[str, int]]:
        """
        Optimize all caches.
        
        Returns:
            Optimization results per cache type
        """
        results = {}
        for cache_type, cache in self.caches.items():
            results[cache_type] = cache.optimize()
        
        logger.info(f"All caches optimized: {results}")
        return results
    
    def get_cache_health(self) -> Dict[str, Any]:
        """
        Get health status of all caches.
        
        Returns:
            Health metrics
        """
        health = {
            'status': 'healthy',
            'caches': {}
        }
        
        for cache_type, cache in self.caches.items():
            stats = cache.get_stats()
            
            cache_health = {
                'utilization': stats.get('utilization', 0),
                'hit_rate': stats.get('hit_rate', 0),
                'size': stats.get('current_size', 0),
                'max_size': stats.get('max_size', 0)
            }
            
            # Determine health status
            if cache_health['utilization'] > 0.9:
                cache_health['status'] = 'warning'
                cache_health['message'] = 'Cache near capacity'
            elif cache_health['hit_rate'] < 0.2 and stats.get('total_requests', 0) > 10:
                cache_health['status'] = 'warning'
                cache_health['message'] = 'Low hit rate'
            else:
                cache_health['status'] = 'healthy'
            
            health['caches'][cache_type] = cache_health
        
        # Overall health
        if any(c.get('status') == 'warning' for c in health['caches'].values()):
            health['status'] = 'warning'
        
        return health

# ============================================
# SINGLETON INSTANCES
# ============================================

# Global multi-cache manager
_cache_manager = None

def get_cache_manager() -> MultiCacheManager:
    """Get singleton cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = MultiCacheManager()
    return _cache_manager

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def get_cache(cache_type: str = 'default') -> Optional[QueryCache]:
    """Get cache instance by type"""
    manager = get_cache_manager()
    return manager.get_cache(cache_type)

def cache_get(key: str, cache_type: str = 'default') -> Optional[Any]:
    """Get value from cache"""
    manager = get_cache_manager()
    return manager.get(key, cache_type)

def cache_set(key: str, value: Any, cache_type: str = 'default', ttl: Optional[int] = None) -> None:
    """Set value in cache"""
    manager = get_cache_manager()
    manager.set(key, value, cache_type, ttl)

def cache_delete(key: str, cache_type: str = 'default') -> bool:
    """Delete from cache"""
    manager = get_cache_manager()
    return manager.delete(key, cache_type)

def get_cache_stats(cache_type: Optional[str] = None) -> Dict[str, Any]:
    """Get cache statistics"""
    manager = get_cache_manager()
    if cache_type:
        cache = manager.get_cache(cache_type)
        return cache.get_stats() if cache else {}
    return manager.get_all_stats()

def optimize_caches() -> Dict[str, Any]:
    """Optimize all caches"""
    manager = get_cache_manager()
    return manager.optimize_all()

# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    print("Testing Enhanced Multi-Type Cache System")
    print("=" * 60)
    
    # Enable granular caching for testing
    FEATURES['granular_caching'] = True
    
    # Get cache manager
    manager = get_cache_manager()
    
    # Test different cache types
    test_data = {
        'final_result': {'query': 'test', 'answer': 'Test answer'},
        'decomposition': {'intent': 'comparison', 'entities': ['DELL', 'IBM']},
        'vendor_resolution': {'input': 'DELL COMPUTER', 'resolved': 'DELL INC'},
        'statistical': {'mean': 1000, 'median': 950, 'std': 100}
    }
    
    print("\n1. Testing cache operations:")
    for cache_type, data in test_data.items():
        key = f"test_key_{cache_type}"
        
        # Set
        manager.set(key, data, cache_type)
        print(f"   Set in {cache_type}: {key}")
        
        # Get
        retrieved = manager.get(key, cache_type)
        print(f"   Retrieved: {retrieved is not None}")
    
    print("\n2. Cache Statistics:")
    stats = manager.get_all_stats()
    for cache_type, cache_stats in stats.items():
        if cache_type != '_aggregate':
            print(f"\n   {cache_type}:")
            print(f"      Size: {cache_stats.get('current_size', 0)}/{cache_stats.get('max_size', 0)}")
            print(f"      Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
            print(f"      TTL: {cache_stats.get('ttl', 0)}s")
    
    print(f"\n   Aggregate:")
    agg = stats.get('_aggregate', {})
    print(f"      Total Entries: {agg.get('total_entries', 0)}")
    print(f"      Overall Hit Rate: {agg.get('overall_hit_rate', 0):.1%}")
    
    print("\n3. Cache Health:")
    health = manager.get_cache_health()
    print(f"   Overall Status: {health['status']}")
    for cache_type, cache_health in health['caches'].items():
        print(f"   {cache_type}: {cache_health.get('status', 'unknown')} "
              f"(utilization: {cache_health.get('utilization', 0):.1%})")
    
    print("\n4. Testing TTL expiration:")
    import time
    
    # Set with short TTL
    manager.set('expiring_key', 'expiring_value', 'final_result', ttl=2)
    print("   Set key with 2s TTL")
    
    # Check immediately
    value = manager.get('expiring_key', 'final_result')
    print(f"   Immediate get: {value is not None}")
    
    # Wait for expiration
    time.sleep(3)
    value = manager.get('expiring_key', 'final_result')
    print(f"   After 3s: {value is not None}")
    
    print("\n5. Cache Optimization:")
    results = manager.optimize_all()
    for cache_type, result in results.items():
        print(f"   {cache_type}: {result['expired_removed']} expired removed")
    
    print("\nEnhanced caching system test complete!")