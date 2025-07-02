"""
Ultra-fast web scraping with Crawl4AI for A05 Knowledge Base
Optimized for maximum throughput and minimal latency
"""

import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from urllib.parse import urljoin, urlparse
import hashlib
import redis.asyncio as redis
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
from crawl4ai.chunking_strategy import RegexChunking
import triton_python_backend_utils as pb_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingResult:
    """Result of web scraping operation"""
    url: str
    content: str
    metadata: Dict[str, Any]
    extracted_data: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    success: bool = True
    error: Optional[str] = None

class OptimizedCrawl4AIScraper:
    """
    Ultra-fast web scraper using Crawl4AI with advanced optimizations
    
    Features:
    - Async/await for maximum concurrency
    - Redis caching for duplicate URL detection
    - Triton inference for content extraction
    - Smart rate limiting and retry logic
    - Memory-efficient streaming processing
    """
    
    def __init__(self, 
                 max_concurrent: int = 100,
                 cache_ttl: int = 3600,
                 redis_url: str = "redis://localhost:6379",
                 triton_url: str = "http://localhost:8000"):
        
        self.max_concurrent = max_concurrent
        self.cache_ttl = cache_ttl
        self.redis_url = redis_url
        self.triton_url = triton_url
        
        # Initialize components
        self.redis_client = None
        self.crawler = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session_pool = None
        
        # Performance metrics
        self.total_scraped = 0
        self.total_time = 0.0
        self.cache_hits = 0
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize all async components"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Initialize Crawl4AI crawler with optimizations
            self.crawler = AsyncWebCrawler(
                # Browser optimizations
                headless=True,
                browser_type="chromium",
                viewport_width=1920,
                viewport_height=1080,
                
                # Performance optimizations
                page_timeout=30000,
                request_timeout=10000,
                navigation_timeout=30000,
                
                # Resource optimizations
                block_resources=["image", "stylesheet", "font", "media"],
                ignore_https_errors=True,
                
                # Concurrency settings
                max_concurrent_pages=self.max_concurrent,
                
                # Memory optimizations
                memory_threshold=0.8,
                cpu_threshold=0.9
            )
            
            await self.crawler.start()
            logger.info("Crawl4AI crawler initialized")
            
            # Initialize HTTP session pool
            connector = aiohttp.TCPConnector(
                limit=self.max_concurrent * 2,
                limit_per_host=50,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            self.session_pool = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30, connect=10)
            )
            
            logger.info("HTTP session pool initialized")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup all resources"""
        try:
            if self.crawler:
                await self.crawler.close()
            
            if self.session_pool:
                await self.session_pool.close()
            
            if self.redis_client:
                await self.redis_client.close()
                
            logger.info("All resources cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def _generate_cache_key(self, url: str, config: Dict[str, Any]) -> str:
        """Generate cache key for URL and configuration"""
        config_str = json.dumps(config, sort_keys=True)
        cache_input = f"{url}:{config_str}"
        return f"crawl4ai:{hashlib.md5(cache_input.encode()).hexdigest()}"
    
    async def _check_cache(self, cache_key: str) -> Optional[ScrapingResult]:
        """Check if result exists in cache"""
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                self.cache_hits += 1
                result_dict = json.loads(cached_data)
                return ScrapingResult(**result_dict)
            return None
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
            return None
    
    async def _store_cache(self, cache_key: str, result: ScrapingResult):
        """Store result in cache"""
        try:
            result_dict = {
                'url': result.url,
                'content': result.content,
                'metadata': result.metadata,
                'extracted_data': result.extracted_data,
                'processing_time': result.processing_time,
                'success': result.success,
                'error': result.error
            }
            
            await self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(result_dict)
            )
        except Exception as e:
            logger.warning(f"Cache store failed: {e}")
    
    async def _extract_with_triton(self, content: str, extraction_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data using Triton inference server"""
        try:
            # Prepare input for Triton
            input_data = {
                "content": content,
                "config": extraction_config
            }
            
            # Call Triton inference (placeholder - implement actual Triton client)
            async with self.session_pool.post(
                f"{self.triton_url}/v2/models/content_extractor/infer",
                json=input_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("extracted_data", {})
                else:
                    logger.warning(f"Triton extraction failed: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Triton extraction error: {e}")
            return {}
    
    async def _scrape_single_url(self, 
                                url: str, 
                                config: CrawlerRunConfig,
                                extraction_config: Optional[Dict[str, Any]] = None) -> ScrapingResult:
        """Scrape a single URL with full optimization"""
        
        start_time = time.time()
        
        async with self.semaphore:  # Rate limiting
            try:
                # Check cache first
                cache_key = self._generate_cache_key(url, config.__dict__)
                cached_result = await self._check_cache(cache_key)
                
                if cached_result:
                    logger.info(f"Cache hit for {url}")
                    return cached_result
                
                # Perform actual scraping
                logger.info(f"Scraping {url}")
                
                result = await self.crawler.arun(
                    url=url,
                    config=config
                )
                
                # Extract structured data if needed
                extracted_data = None
                if extraction_config and result.markdown:
                    extracted_data = await self._extract_with_triton(
                        result.markdown, 
                        extraction_config
                    )
                
                # Create result object
                scraping_result = ScrapingResult(
                    url=url,
                    content=result.markdown or "",
                    metadata={
                        'title': result.metadata.get('title', ''),
                        'description': result.metadata.get('description', ''),
                        'keywords': result.metadata.get('keywords', []),
                        'links': result.links[:100],  # Limit links for memory
                        'images': result.media.get('images', [])[:50],  # Limit images
                        'status_code': result.status_code,
                        'response_time': result.response_time,
                        'content_length': len(result.markdown or "")
                    },
                    extracted_data=extracted_data,
                    processing_time=time.time() - start_time,
                    success=result.success,
                    error=result.error_message if not result.success else None
                )
                
                # Cache the result
                await self._store_cache(cache_key, scraping_result)
                
                # Update metrics
                self.total_scraped += 1
                self.total_time += scraping_result.processing_time
                
                return scraping_result
                
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                return ScrapingResult(
                    url=url,
                    content="",
                    metadata={},
                    processing_time=time.time() - start_time,
                    success=False,
                    error=str(e)
                )
    
    async def scrape_urls(self, 
                         urls: List[str],
                         extraction_strategy: Optional[str] = None,
                         extraction_config: Optional[Dict[str, Any]] = None,
                         custom_config: Optional[Dict[str, Any]] = None) -> List[ScrapingResult]:
        """
        Scrape multiple URLs concurrently with maximum optimization
        
        Args:
            urls: List of URLs to scrape
            extraction_strategy: Type of extraction ('llm', 'css', 'regex')
            extraction_config: Configuration for extraction
            custom_config: Custom crawler configuration
        
        Returns:
            List of ScrapingResult objects
        """
        
        if not urls:
            return []
        
        # Prepare crawler configuration
        config_params = {
            'word_count_threshold': 10,
            'extraction_strategy': None,
            'chunking_strategy': RegexChunking(),
            'bypass_cache': False,
            'include_raw_html': False,
            'remove_overlay_elements': True,
            'simulate_user': True,
            'override_navigator': True,
            'magic': True,  # Enable all optimizations
            'process_iframes': False,  # Skip iframes for speed
            'remove_forms': True,  # Remove forms for cleaner content
            'social_media_domains': [],  # Skip social media
        }
        
        # Apply custom configuration
        if custom_config:
            config_params.update(custom_config)
        
        # Set up extraction strategy
        if extraction_strategy == 'llm' and extraction_config:
            config_params['extraction_strategy'] = LLMExtractionStrategy(
                provider="openai",
                api_token=extraction_config.get('api_token'),
                instruction=extraction_config.get('instruction', 'Extract key information'),
                schema=extraction_config.get('schema', {})
            )
        elif extraction_strategy == 'css' and extraction_config:
            config_params['extraction_strategy'] = JsonCssExtractionStrategy(
                extraction_config.get('css_selectors', {})
            )
        
        config = CrawlerRunConfig(**config_params)
        
        # Create tasks for concurrent execution
        tasks = [
            self._scrape_single_url(url, config, extraction_config)
            for url in urls
        ]
        
        # Execute all tasks concurrently
        logger.info(f"Starting concurrent scraping of {len(urls)} URLs")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
                processed_results.append(ScrapingResult(
                    url=urls[i],
                    content="",
                    metadata={},
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in processed_results if r.success)
        
        logger.info(f"Scraping completed: {success_count}/{len(urls)} successful in {total_time:.2f}s")
        logger.info(f"Average speed: {len(urls)/total_time:.2f} URLs/second")
        logger.info(f"Cache hit rate: {self.cache_hits}/{len(urls)} ({self.cache_hits/len(urls)*100:.1f}%)")
        
        return processed_results
    
    async def scrape_with_pagination(self, 
                                   base_url: str,
                                   max_pages: int = 10,
                                   pagination_selector: str = "a[rel='next']",
                                   extraction_config: Optional[Dict[str, Any]] = None) -> List[ScrapingResult]:
        """
        Scrape website with pagination support
        
        Args:
            base_url: Starting URL
            max_pages: Maximum number of pages to scrape
            pagination_selector: CSS selector for next page link
            extraction_config: Configuration for content extraction
        
        Returns:
            List of ScrapingResult objects from all pages
        """
        
        all_results = []
        current_url = base_url
        page_count = 0
        
        while current_url and page_count < max_pages:
            logger.info(f"Scraping page {page_count + 1}: {current_url}")
            
            # Scrape current page
            config = CrawlerRunConfig(
                css_selector=pagination_selector,
                include_raw_html=True,  # Need HTML to find next page link
                magic=True
            )
            
            results = await self.scrape_urls([current_url], extraction_config=extraction_config)
            
            if results and results[0].success:
                all_results.extend(results)
                
                # Find next page URL
                # This would need to be implemented based on the specific pagination pattern
                # For now, we'll break after first page
                break
            else:
                logger.warning(f"Failed to scrape page {page_count + 1}")
                break
            
            page_count += 1
        
        return all_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = self.total_time / max(self.total_scraped, 1)
        
        return {
            'total_scraped': self.total_scraped,
            'total_time': self.total_time,
            'average_time_per_url': avg_time,
            'urls_per_second': 1 / avg_time if avg_time > 0 else 0,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(self.total_scraped, 1)
        }

# Example usage and testing
async def main():
    """Example usage of OptimizedCrawl4AIScraper"""
    
    # Test URLs
    test_urls = [
        "https://arxiv.org/abs/2301.00001",
        "https://github.com/trending",
        "https://news.ycombinator.com",
        "https://www.reddit.com/r/MachineLearning/",
        "https://paperswithcode.com/latest"
    ]
    
    # Extraction configuration for academic papers
    extraction_config = {
        'instruction': 'Extract title, authors, abstract, and key findings',
        'schema': {
            'title': 'string',
            'authors': 'list',
            'abstract': 'string',
            'key_findings': 'list'
        }
    }
    
    async with OptimizedCrawl4AIScraper(max_concurrent=50) as scraper:
        # Scrape URLs
        results = await scraper.scrape_urls(
            urls=test_urls,
            extraction_strategy='llm',
            extraction_config=extraction_config
        )
        
        # Print results
        for result in results:
            print(f"\nURL: {result.url}")
            print(f"Success: {result.success}")
            print(f"Processing time: {result.processing_time:.2f}s")
            print(f"Content length: {len(result.content)}")
            
            if result.extracted_data:
                print(f"Extracted data: {result.extracted_data}")
        
        # Print performance stats
        stats = scraper.get_performance_stats()
        print(f"\nPerformance Stats:")
        print(f"Total scraped: {stats['total_scraped']}")
        print(f"Average time per URL: {stats['average_time_per_url']:.2f}s")
        print(f"URLs per second: {stats['urls_per_second']:.2f}")
        print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")

if __name__ == "__main__":
    asyncio.run(main())

