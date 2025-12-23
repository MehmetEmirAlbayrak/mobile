"""
QoS (Quality of Service) Optimization Module
Provides QoS recommendations based on traffic classification
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class Priority(Enum):
    """Traffic priority levels for QoS"""
    CRITICAL = 1      # Real-time, delay-sensitive (VoIP)
    HIGH = 2          # Near real-time (Video streaming)
    MEDIUM = 3        # Interactive (Web browsing, Chat)
    LOW = 4           # Bulk transfer (File transfer, Email)
    BEST_EFFORT = 5   # Background (P2P)

@dataclass
class QoSProfile:
    """QoS profile for a traffic class"""
    traffic_class: str
    priority: Priority
    min_bandwidth_kbps: int      # Minimum guaranteed bandwidth
    max_bandwidth_kbps: int      # Maximum allowed bandwidth
    max_latency_ms: int          # Maximum acceptable latency
    max_jitter_ms: int           # Maximum acceptable jitter
    packet_loss_tolerance: float # Acceptable packet loss percentage
    dscp_value: int              # Differentiated Services Code Point
    queue_weight: int            # Weight for weighted fair queuing (1-100)
    description: str

# QoS Profiles for each traffic class
QOS_PROFILES: Dict[str, QoSProfile] = {
    'VOIP': QoSProfile(
        traffic_class='VOIP',
        priority=Priority.CRITICAL,
        min_bandwidth_kbps=100,
        max_bandwidth_kbps=500,
        max_latency_ms=150,
        max_jitter_ms=30,
        packet_loss_tolerance=0.01,
        dscp_value=46,  # EF (Expedited Forwarding)
        queue_weight=100,
        description='Voice over IP - Requires lowest latency and jitter for clear calls'
    ),
    'STREAMING': QoSProfile(
        traffic_class='STREAMING',
        priority=Priority.HIGH,
        min_bandwidth_kbps=2000,
        max_bandwidth_kbps=25000,
        max_latency_ms=300,
        max_jitter_ms=50,
        packet_loss_tolerance=0.05,
        dscp_value=34,  # AF41 (Assured Forwarding)
        queue_weight=80,
        description='Video/Audio Streaming - High bandwidth, moderate latency tolerance'
    ),
    'CHAT': QoSProfile(
        traffic_class='CHAT',
        priority=Priority.MEDIUM,
        min_bandwidth_kbps=50,
        max_bandwidth_kbps=500,
        max_latency_ms=500,
        max_jitter_ms=100,
        packet_loss_tolerance=0.01,
        dscp_value=26,  # AF31
        queue_weight=60,
        description='Chat/Messaging - Low bandwidth, moderate latency for responsiveness'
    ),
    'BROWSING': QoSProfile(
        traffic_class='BROWSING',
        priority=Priority.MEDIUM,
        min_bandwidth_kbps=500,
        max_bandwidth_kbps=10000,
        max_latency_ms=500,
        max_jitter_ms=200,
        packet_loss_tolerance=0.02,
        dscp_value=18,  # AF21
        queue_weight=50,
        description='Web Browsing - Variable bandwidth, user expects quick page loads'
    ),
    'MAIL': QoSProfile(
        traffic_class='MAIL',
        priority=Priority.LOW,
        min_bandwidth_kbps=100,
        max_bandwidth_kbps=5000,
        max_latency_ms=2000,
        max_jitter_ms=500,
        packet_loss_tolerance=0.01,
        dscp_value=10,  # AF11
        queue_weight=30,
        description='Email Traffic - Delay tolerant, reliability more important than speed'
    ),
    'FILE_TRANSFER': QoSProfile(
        traffic_class='FILE_TRANSFER',
        priority=Priority.LOW,
        min_bandwidth_kbps=500,
        max_bandwidth_kbps=50000,
        max_latency_ms=5000,
        max_jitter_ms=1000,
        packet_loss_tolerance=0.001,
        dscp_value=10,  # AF11
        queue_weight=25,
        description='File Transfer - High bandwidth when available, very delay tolerant'
    ),
    'P2P': QoSProfile(
        traffic_class='P2P',
        priority=Priority.BEST_EFFORT,
        min_bandwidth_kbps=100,
        max_bandwidth_kbps=10000,
        max_latency_ms=10000,
        max_jitter_ms=2000,
        packet_loss_tolerance=0.05,
        dscp_value=0,   # Best Effort
        queue_weight=10,
        description='Peer-to-Peer - Lowest priority, uses remaining bandwidth'
    )
}

class QoSOptimizer:
    """
    QoS Optimizer that provides recommendations based on traffic classification
    """
    
    def __init__(self, total_bandwidth_mbps: float = 100.0):
        """
        Initialize QoS Optimizer
        
        Args:
            total_bandwidth_mbps: Total available bandwidth in Mbps
        """
        self.total_bandwidth_kbps = total_bandwidth_mbps * 1000
        self.profiles = QOS_PROFILES
        self.active_flows: Dict[str, int] = {}  # class -> count
    
    def get_qos_recommendation(self, traffic_class: str) -> Dict:
        """
        Get QoS recommendation for a traffic class
        
        Args:
            traffic_class: Classified traffic type
            
        Returns:
            Dictionary with QoS parameters and recommendations
        """
        profile = self.profiles.get(traffic_class)
        
        if not profile:
            return {
                'error': f'Unknown traffic class: {traffic_class}',
                'recommendation': 'Apply best-effort QoS'
            }
        
        return {
            'traffic_class': profile.traffic_class,
            'priority': {
                'level': profile.priority.value,
                'name': profile.priority.name,
                'color': self._get_priority_color(profile.priority)
            },
            'bandwidth': {
                'min_kbps': profile.min_bandwidth_kbps,
                'max_kbps': profile.max_bandwidth_kbps,
                'min_display': self._format_bandwidth(profile.min_bandwidth_kbps),
                'max_display': self._format_bandwidth(profile.max_bandwidth_kbps)
            },
            'latency': {
                'max_ms': profile.max_latency_ms,
                'display': f'< {profile.max_latency_ms}ms'
            },
            'jitter': {
                'max_ms': profile.max_jitter_ms,
                'display': f'< {profile.max_jitter_ms}ms'
            },
            'packet_loss': {
                'tolerance': profile.packet_loss_tolerance,
                'display': f'< {profile.packet_loss_tolerance * 100:.1f}%'
            },
            'dscp': {
                'value': profile.dscp_value,
                'binary': format(profile.dscp_value, '06b'),
                'hex': hex(profile.dscp_value)
            },
            'queue_weight': profile.queue_weight,
            'description': profile.description,
            'actions': self._get_recommended_actions(profile)
        }
    
    def allocate_bandwidth(self, traffic_counts: Dict[str, int]) -> Dict:
        """
        Allocate bandwidth based on active traffic flows
        
        Args:
            traffic_counts: Dictionary of traffic class -> number of flows
            
        Returns:
            Bandwidth allocation for each class
        """
        self.active_flows = traffic_counts
        total_weight = 0
        allocations = {}
        
        # Calculate total weight
        for traffic_class, count in traffic_counts.items():
            if traffic_class in self.profiles and count > 0:
                profile = self.profiles[traffic_class]
                total_weight += profile.queue_weight * count
        
        if total_weight == 0:
            return {'error': 'No active flows'}
        
        # Allocate bandwidth proportionally
        for traffic_class, count in traffic_counts.items():
            if traffic_class in self.profiles and count > 0:
                profile = self.profiles[traffic_class]
                weight_share = (profile.queue_weight * count) / total_weight
                allocated_kbps = self.total_bandwidth_kbps * weight_share
                
                # Ensure minimum bandwidth
                allocated_kbps = max(allocated_kbps, profile.min_bandwidth_kbps * count)
                # Cap at maximum
                allocated_kbps = min(allocated_kbps, profile.max_bandwidth_kbps * count)
                
                allocations[traffic_class] = {
                    'flows': count,
                    'total_kbps': int(allocated_kbps),
                    'per_flow_kbps': int(allocated_kbps / count),
                    'display': self._format_bandwidth(allocated_kbps),
                    'percentage': round(allocated_kbps / self.total_bandwidth_kbps * 100, 1),
                    'priority': profile.priority.name
                }
        
        return {
            'total_bandwidth': self._format_bandwidth(self.total_bandwidth_kbps),
            'allocations': allocations
        }
    
    def get_priority_queue(self, traffic_classes: List[str]) -> List[Dict]:
        """
        Get priority-ordered queue for given traffic classes
        
        Args:
            traffic_classes: List of traffic classes to queue
            
        Returns:
            Sorted list of traffic with priority info
        """
        queue = []
        
        for traffic_class in traffic_classes:
            if traffic_class in self.profiles:
                profile = self.profiles[traffic_class]
                queue.append({
                    'traffic_class': traffic_class,
                    'priority': profile.priority.value,
                    'priority_name': profile.priority.name,
                    'color': self._get_priority_color(profile.priority),
                    'queue_weight': profile.queue_weight
                })
        
        # Sort by priority (lower value = higher priority)
        queue.sort(key=lambda x: x['priority'])
        
        return queue
    
    def _get_priority_color(self, priority: Priority) -> str:
        """Get color for priority level"""
        colors = {
            Priority.CRITICAL: '#ef4444',    # Red
            Priority.HIGH: '#f97316',        # Orange
            Priority.MEDIUM: '#eab308',      # Yellow
            Priority.LOW: '#22c55e',         # Green
            Priority.BEST_EFFORT: '#6b7280'  # Gray
        }
        return colors.get(priority, '#6b7280')
    
    def _format_bandwidth(self, kbps: float) -> str:
        """Format bandwidth for display"""
        if kbps >= 1000:
            return f'{kbps/1000:.1f} Mbps'
        return f'{kbps:.0f} Kbps'
    
    def _get_recommended_actions(self, profile: QoSProfile) -> List[str]:
        """Get recommended network actions for a traffic class"""
        actions = []
        
        if profile.priority == Priority.CRITICAL:
            actions.append('🔴 Place in priority queue (strict priority)')
            actions.append('🚀 Enable low-latency mode')
            actions.append('🛡️ Reserve dedicated bandwidth')
        elif profile.priority == Priority.HIGH:
            actions.append('🟠 Place in high-priority queue')
            actions.append('📊 Enable adaptive bitrate support')
            actions.append('💾 Allocate buffer for jitter compensation')
        elif profile.priority == Priority.MEDIUM:
            actions.append('🟡 Place in standard queue with weight')
            actions.append('⚖️ Apply fair queuing')
        elif profile.priority == Priority.LOW:
            actions.append('🟢 Place in bulk transfer queue')
            actions.append('⏰ Allow scheduling during off-peak')
        else:
            actions.append('⚪ Place in best-effort queue')
            actions.append('🔽 Apply rate limiting if congested')
        
        actions.append(f'🏷️ Mark packets with DSCP {profile.dscp_value}')
        
        return actions


# Singleton instance
qos_optimizer = QoSOptimizer()

def get_qos_for_traffic(traffic_class: str) -> Dict:
    """Convenience function to get QoS recommendation"""
    return qos_optimizer.get_qos_recommendation(traffic_class)

def allocate_bandwidth_for_traffic(traffic_counts: Dict[str, int]) -> Dict:
    """Convenience function to allocate bandwidth"""
    return qos_optimizer.allocate_bandwidth(traffic_counts)

