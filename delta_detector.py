"""
Transcript Delta Detector
Compares two earnings call transcripts and detects material changes.
"""

import difflib
import re
from typing import List, Dict, Tuple
import json
from sentiment_analyzer import FinancialSentimentAnalyzer

class DeltaDetector:
    """Detect and classify changes between two earnings call transcripts."""
    
    def __init__(self):
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        
        # Topics to track
        self.topics = [
            'revenue', 'guidance', 'margin', 'demand', 
            'competitive', 'regulatory', 'cost', 'growth'
        ]
        
        # Keywords for topic detection
        self.topic_keywords = {
            'revenue': ['revenue', 'sales', 'top line', 'income'],
            'guidance': ['guidance', 'outlook', 'expect', 'forecast', 'project'],
            'margin': ['margin', 'gross margin', 'operating margin', 'profitability'],
            'demand': ['demand', 'customer', 'orders', 'pipeline', 'backlog'],
            'competitive': ['competitive', 'competition', 'market share', 'rival'],
            'regulatory': ['regulatory', 'regulation', 'compliance', 'legal'],
            'cost': ['cost', 'expense', 'spending', 'inflation', 'supply chain'],
            'growth': ['growth', 'expansion', 'momentum', 'accelerat']
        }
    
    def extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def detect_changes(self, current_text: str, previous_text: str) -> Dict:
        """
        Detect changes between two transcript texts.
        
        Returns:
            Dictionary with added, removed, and modified passages
        """
        current_sentences = self.extract_sentences(current_text)
        previous_sentences = self.extract_sentences(previous_text)
        
        # Use difflib to find differences
        matcher = difflib.SequenceMatcher(None, previous_sentences, current_sentences)
        
        added = []
        removed = []
        modified = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'insert':
                added.extend(current_sentences[j1:j2])
            elif tag == 'delete':
                removed.extend(previous_sentences[i1:i2])
            elif tag == 'replace':
                # Potential modification
                for prev, curr in zip(previous_sentences[i1:i2], current_sentences[j1:j2]):
                    if self._is_substantive_change(prev, curr):
                        modified.append({
                            'previous': prev,
                            'current': curr
                        })
        
        return {
            'added_count': len(added),
            'removed_count': len(removed),
            'modified_count': len(modified),
            'added_examples': added[:3],
            'removed_examples': removed[:3],
            'modified_examples': modified[:3]
        }
    
    def _is_substantive_change(self, prev: str, curr: str) -> bool:
        """Determine if a change is substantive (not just wording)."""
        # Simple heuristic: if length changes by >20% or key words change
        len_ratio = len(curr) / max(len(prev), 1)
        
        # Check for sentiment change
        prev_sentiment = self.sentiment_analyzer.analyze(prev)
        curr_sentiment = self.sentiment_analyzer.analyze(curr)
        
        sentiment_change = abs(prev_sentiment['sentiment_score'] - curr_sentiment['sentiment_score'])
        
        return (len_ratio > 1.2 or len_ratio < 0.8) or sentiment_change > 0.15
    
    def classify_by_topic(self, changes: Dict) -> Dict:
        """Classify detected changes by topic."""
        topic_changes = {topic: [] for topic in self.topics}
        
        # Check added sentences
        for sentence in changes.get('added_examples', []):
            topic = self._detect_topic(sentence)
            if topic:
                topic_changes[topic].append({
                    'type': 'added',
                    'text': sentence
                })
        
        # Check modified sentences
        for mod in changes.get('modified_examples', []):
            topic = self._detect_topic(mod['current'])
            if topic:
                topic_changes[topic].append({
                    'type': 'modified',
                    'previous': mod['previous'],
                    'current': mod['current']
                })
        
        # Add sentiment analysis for topics with changes
        result = {}
        for topic, changes_list in topic_changes.items():
            if changes_list:
                result[topic] = {
                    'change_count': len(changes_list),
                    'changes': changes_list[:2],  # First two examples
                    'materiality': self._assess_materiality(topic, changes_list)
                }
        
        return result
    
    def _detect_topic(self, text: str) -> str:
        """Detect which topic a sentence belongs to."""
        text_lower = text.lower()
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return topic
        return None
    
    def _assess_materiality(self, topic: str, changes: List) -> str:
        """Assess materiality of changes for a topic."""
        # High materiality topics
        high_impact_topics = ['guidance', 'margin', 'revenue']
        
        if topic in high_impact_topics and len(changes) >= 1:
            return 'high'
        elif len(changes) >= 2:
            return 'moderate'
        else:
            return 'low'
    
    def generate_delta_report(self, current_text: str, previous_text: str, 
                             company: str, current_quarter: str, previous_quarter: str) -> Dict:
        """
        Generate complete delta report for two transcripts.
        """
        # Detect changes
        changes = self.detect_changes(current_text, previous_text)
        
        # Classify by topic
        topic_analysis = self.classify_by_topic(changes)
        
        # Analyze sentiment shifts for key topics
        sentiment_shifts = {}
        for topic in self.topics[:4]:  # Top 4 topics for v1
            try:
                shift = self.sentiment_analyzer.compare_sentiment(current_text, previous_text, topic)
                if 'error' not in shift:
                    sentiment_shifts[topic] = shift
            except Exception:
                pass
        
        return {
            'company': company,
            'current_quarter': current_quarter,
            'previous_quarter': previous_quarter,
            'summary': {
                'total_added': changes['added_count'],
                'total_removed': changes['removed_count'],
                'total_modified': changes['modified_count'],
                'has_material_changes': len(topic_analysis) > 0
            },
            'topic_changes': topic_analysis,
            'sentiment_shifts': sentiment_shifts,
            'top_change': self._get_top_change(topic_analysis, sentiment_shifts)
        }
    
    def _get_top_change(self, topic_analysis: Dict, sentiment_shifts: Dict) -> Dict:
        """Identify the most material change."""
        # Prioritize guidance and margin changes with high materiality
        for topic in ['guidance', 'margin', 'revenue']:
            if topic in topic_analysis and topic_analysis[topic].get('materiality') == 'high':
                return {
                    'topic': topic,
                    'type': 'language_and_sentiment',
                    'summary': f"Material changes detected in {topic} language with high materiality"
                }
        
        if sentiment_shifts:
            top_topic = max(sentiment_shifts.items(), key=lambda x: abs(x[1].get('sentiment_shift', 0)))
            return {
                'topic': top_topic[0],
                'type': 'sentiment_shift',
                'shift': top_topic[1]['sentiment_shift'],
                'direction': top_topic[1]['direction']
            }
        
        return {'message': 'No material changes detected'}


# Example usage
if __name__ == "__main__":
    detector = DeltaDetector()
    
    # Example texts
    current = "We expect revenue to grow 15% next quarter. Margins remain strong. Competition is increasing but we are well positioned."
    previous = "Revenue growth will be modest next quarter. Margins are under pressure. Competition is a concern."
    
    report = detector.generate_delta_report(current, previous, "NVDA", "Q3 2024", "Q2 2024")
    print(json.dumps(report, indent=2))
