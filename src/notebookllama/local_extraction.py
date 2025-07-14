import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class SourceText:
    """Represents source text for extraction."""
    text_content: str
    filename: str


class LocalExtractor:
    """Local extractor for structured data from text."""
    
    def __init__(self):
        """Initialize the local extractor."""
        pass
    
    async def aextract(self, files: SourceText) -> "ExtractionOutput":
        """Async extract structured data from text."""
        return self.extract(files)
    
    def extract(self, files: SourceText) -> "ExtractionOutput":
        """Extract structured data from text."""
        text = files.text_content
        
        # Extract basic information
        extraction_data = {
            "summary": self._extract_summary(text),
            "key_points": self._extract_key_points(text),
            "entities": self._extract_entities(text),
            "topics": self._extract_topics(text),
            "sentiment": self._extract_sentiment(text),
            "metadata": {
                "filename": files.filename,
                "word_count": len(text.split()),
                "character_count": len(text),
            }
        }
        
        return ExtractionOutput(data=extraction_data)
    
    def _extract_summary(self, text: str) -> str:
        """Extract a summary from the text."""
        # Simple approach: take first few sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            return text
        
        # Take first 3 sentences as summary
        summary_sentences = sentences[:3]
        return '. '.join(summary_sentences) + '.'
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from the text."""
        # Look for bullet points, numbered lists, or sentences with key phrases
        key_points = []
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Look for bullet points
            if paragraph.startswith(('•', '-', '*', '1.', '2.', '3.')):
                key_points.append(paragraph)
            # Look for sentences with key phrases
            elif any(phrase in paragraph.lower() for phrase in [
                'important', 'key', 'main', 'primary', 'essential', 'critical'
            ]):
                key_points.append(paragraph)
            # Take short, impactful sentences
            elif len(paragraph.split()) <= 20 and len(paragraph) > 50:
                key_points.append(paragraph)
        
        # Limit to top 5 key points
        return key_points[:5]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from the text."""
        # Simple entity extraction using regex patterns
        entities = []
        
        # Extract capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.extend(capitalized_words)
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        entities.extend(emails)
        
        # Extract URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        entities.extend(urls)
        
        # Remove duplicates and limit
        unique_entities = list(set(entities))
        return unique_entities[:10]
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from the text."""
        # Simple topic extraction based on frequency
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count word frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words as topics
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [topic[0] for topic in topics[:5]]
    
    def _extract_sentiment(self, text: str) -> str:
        """Extract sentiment from the text."""
        # Simple sentiment analysis based on positive/negative words
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'positive', 'happy', 'successful', 'beneficial', 'advantageous'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'negative', 'sad',
            'unsuccessful', 'harmful', 'disadvantageous', 'problem', 'issue'
        }
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"


class ExtractionOutput:
    """Represents extraction output."""
    
    def __init__(self, data: Dict[str, Any]):
        """Initialize extraction output."""
        self.data = data 