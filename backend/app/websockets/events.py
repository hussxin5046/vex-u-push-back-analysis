"""
WebSocket event handlers for real-time communication
"""

from flask_socketio import emit, join_room, leave_room, disconnect
from flask import request
from app import socketio
import logging
import json
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# Store active connections and subscriptions
active_connections = {}
room_subscriptions = {}

@socketio.event
def connect():
    """Handle client connection"""
    try:
        client_id = str(uuid.uuid4())
        active_connections[request.sid] = {
            "client_id": client_id,
            "connected_at": datetime.utcnow(),
            "subscriptions": []
        }
        
        logger.info(f"Client connected: {client_id} (session: {request.sid})")
        
        emit('connected', {
            "client_id": client_id,
            "server_time": datetime.utcnow().isoformat(),
            "message": "Connected to VEX U Analysis API WebSocket"
        })
        
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
        emit('error', {"message": "Connection failed", "error": str(e)})

@socketio.event
def disconnect():
    """Handle client disconnection"""
    try:
        if request.sid in active_connections:
            client_info = active_connections[request.sid]
            client_id = client_info["client_id"]
            
            # Remove from all rooms
            for subscription in client_info["subscriptions"]:
                leave_room(subscription)
                if subscription in room_subscriptions:
                    room_subscriptions[subscription].discard(request.sid)
            
            # Remove from active connections
            del active_connections[request.sid]
            
            logger.info(f"Client disconnected: {client_id} (session: {request.sid})")
        
    except Exception as e:
        logger.error(f"Disconnection error: {str(e)}")

@socketio.event
def subscribe(data):
    """Subscribe to specific data streams"""
    try:
        if request.sid not in active_connections:
            emit('error', {"message": "Not connected"})
            return
        
        subscription_type = data.get('type')
        params = data.get('params', {})
        
        if not subscription_type:
            emit('error', {"message": "Subscription type required"})
            return
        
        # Create room name
        room_name = f"{subscription_type}_{params.get('id', 'general')}"
        
        # Join room
        join_room(room_name)
        
        # Track subscription
        active_connections[request.sid]["subscriptions"].append(room_name)
        
        if room_name not in room_subscriptions:
            room_subscriptions[room_name] = set()
        room_subscriptions[room_name].add(request.sid)
        
        logger.info(f"Client {request.sid} subscribed to {room_name}")
        
        emit('subscribed', {
            "type": subscription_type,
            "room": room_name,
            "params": params,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Send initial data based on subscription type
        if subscription_type == "analysis_progress":
            emit('analysis_update', {
                "status": "ready",
                "message": "Ready to receive analysis updates",
                "timestamp": datetime.utcnow().isoformat()
            })
        elif subscription_type == "ml_training":
            emit('training_update', {
                "status": "monitoring",
                "message": "Ready to receive training updates", 
                "timestamp": datetime.utcnow().isoformat()
            })
        elif subscription_type == "system_metrics":
            emit('metrics_update', {
                "status": "monitoring",
                "message": "Ready to receive system metrics",
                "timestamp": datetime.utcnow().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Subscription error: {str(e)}")
        emit('error', {"message": "Subscription failed", "error": str(e)})

@socketio.event
def unsubscribe(data):
    """Unsubscribe from data streams"""
    try:
        if request.sid not in active_connections:
            emit('error', {"message": "Not connected"})
            return
        
        subscription_type = data.get('type')
        params = data.get('params', {})
        
        room_name = f"{subscription_type}_{params.get('id', 'general')}"
        
        # Leave room
        leave_room(room_name)
        
        # Remove from tracking
        if room_name in active_connections[request.sid]["subscriptions"]:
            active_connections[request.sid]["subscriptions"].remove(room_name)
        
        if room_name in room_subscriptions:
            room_subscriptions[room_name].discard(request.sid)
        
        logger.info(f"Client {request.sid} unsubscribed from {room_name}")
        
        emit('unsubscribed', {
            "type": subscription_type,
            "room": room_name,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Unsubscription error: {str(e)}")
        emit('error', {"message": "Unsubscription failed", "error": str(e)})

@socketio.event
def ping():
    """Handle ping for keepalive"""
    emit('pong', {"timestamp": datetime.utcnow().isoformat()})

@socketio.event
def get_status():
    """Get connection status"""
    try:
        if request.sid in active_connections:
            client_info = active_connections[request.sid]
            emit('status', {
                "connected": True,
                "client_id": client_info["client_id"],
                "connected_at": client_info["connected_at"].isoformat(),
                "subscriptions": client_info["subscriptions"],
                "timestamp": datetime.utcnow().isoformat()
            })
        else:
            emit('status', {
                "connected": False,
                "timestamp": datetime.utcnow().isoformat()
            })
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        emit('error', {"message": "Status check failed", "error": str(e)})

# Utility functions for broadcasting updates

def broadcast_analysis_progress(analysis_id, status, progress, message=None):
    """Broadcast analysis progress to subscribers"""
    try:
        room_name = f"analysis_progress_{analysis_id}"
        
        update_data = {
            "analysis_id": analysis_id,
            "status": status,
            "progress": progress,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        socketio.emit('analysis_update', update_data, room=room_name)
        logger.info(f"Broadcasted analysis progress for {analysis_id}: {progress}%")
        
    except Exception as e:
        logger.error(f"Failed to broadcast analysis progress: {str(e)}")

def broadcast_ml_training_progress(job_id, model_type, epoch, total_epochs, metrics=None):
    """Broadcast ML training progress to subscribers"""
    try:
        room_name = f"ml_training_{job_id}"
        
        update_data = {
            "job_id": job_id,
            "model_type": model_type,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "progress": (epoch / total_epochs) * 100 if total_epochs > 0 else 0,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        socketio.emit('training_update', update_data, room=room_name)
        logger.info(f"Broadcasted training progress for {job_id}: epoch {epoch}/{total_epochs}")
        
    except Exception as e:
        logger.error(f"Failed to broadcast training progress: {str(e)}")

def broadcast_system_metrics(metrics):
    """Broadcast system metrics to subscribers"""
    try:
        room_name = "system_metrics_general"
        
        update_data = {
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        socketio.emit('metrics_update', update_data, room=room_name)
        logger.debug("Broadcasted system metrics")
        
    except Exception as e:
        logger.error(f"Failed to broadcast system metrics: {str(e)}")

def broadcast_notification(notification_type, title, message, data=None):
    """Broadcast notifications to all connected clients"""
    try:
        notification_data = {
            "type": notification_type,
            "title": title,
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        socketio.emit('notification', notification_data)
        logger.info(f"Broadcasted notification: {title}")
        
    except Exception as e:
        logger.error(f"Failed to broadcast notification: {str(e)}")

def broadcast_visualization_update(viz_id, chart_data):
    """Broadcast visualization updates to subscribers"""
    try:
        room_name = f"visualization_{viz_id}"
        
        update_data = {
            "visualization_id": viz_id,
            "chart_data": chart_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        socketio.emit('visualization_update', update_data, room=room_name)
        logger.info(f"Broadcasted visualization update for {viz_id}")
        
    except Exception as e:
        logger.error(f"Failed to broadcast visualization update: {str(e)}")

def get_connection_stats():
    """Get WebSocket connection statistics"""
    try:
        stats = {
            "total_connections": len(active_connections),
            "total_rooms": len(room_subscriptions),
            "connections_by_room": {
                room: len(sessions) for room, sessions in room_subscriptions.items()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        return stats
    except Exception as e:
        logger.error(f"Failed to get connection stats: {str(e)}")
        return {"error": str(e)}