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
            "message": "Connected to Push Back Strategy WebSocket"
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
        if subscription_type == "push_back_analysis":
            emit('analysis_update', {
                "status": "ready",
                "message": "Ready to receive Push Back analysis updates",
                "timestamp": datetime.utcnow().isoformat()
            })
        elif subscription_type == "strategy_optimization":
            emit('optimization_update', {
                "status": "monitoring",
                "message": "Ready to receive strategy optimization updates",
                "timestamp": datetime.utcnow().isoformat()
            })
        elif subscription_type == "monte_carlo_simulation":
            emit('simulation_update', {
                "status": "monitoring", 
                "message": "Ready to receive Monte Carlo simulation updates",
                "timestamp": datetime.utcnow().isoformat()
            })
        elif subscription_type == "field_state":
            emit('field_update', {
                "status": "monitoring",
                "message": "Ready to receive field state updates",
                "timestamp": datetime.utcnow().isoformat()
            })
        elif subscription_type == "match_scoring":
            emit('score_update', {
                "status": "monitoring",
                "message": "Ready to receive scoring updates",
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

# Push Back Specific WebSocket Functions

def broadcast_push_back_analysis_progress(analysis_id, stage, progress, current_step=None, results=None):
    """Broadcast Push Back analysis progress"""
    try:
        room_name = f"push_back_analysis_{analysis_id}"
        
        update_data = {
            "analysis_id": analysis_id,
            "stage": stage,  # "block_flow", "autonomous_decision", "goal_priority", etc.
            "progress": progress,  # 0-100
            "current_step": current_step,
            "partial_results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        socketio.emit('push_back_analysis_update', update_data, room=room_name)
        logger.info(f"Broadcasted Push Back analysis progress for {analysis_id}: {stage} {progress}%")
        
    except Exception as e:
        logger.error(f"Failed to broadcast Push Back analysis progress: {str(e)}")

def broadcast_strategy_optimization_update(strategy_id, optimization_type, progress, current_values=None):
    """Broadcast strategy optimization progress"""
    try:
        room_name = f"strategy_optimization_{strategy_id}"
        
        update_data = {
            "strategy_id": strategy_id,
            "optimization_type": optimization_type,  # "block_flow", "parking", "autonomous"
            "progress": progress,
            "current_values": current_values,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        socketio.emit('strategy_optimization_update', update_data, room=room_name)
        logger.info(f"Broadcasted optimization update for {strategy_id}: {optimization_type} {progress}%")
        
    except Exception as e:
        logger.error(f"Failed to broadcast optimization update: {str(e)}")

def broadcast_monte_carlo_progress(simulation_id, completed_sims, total_sims, current_stats=None):
    """Broadcast Monte Carlo simulation progress"""
    try:
        room_name = f"monte_carlo_simulation_{simulation_id}"
        
        progress = (completed_sims / total_sims) * 100 if total_sims > 0 else 0
        
        update_data = {
            "simulation_id": simulation_id,
            "completed_simulations": completed_sims,
            "total_simulations": total_sims,
            "progress": progress,
            "current_stats": current_stats,  # Win rate, avg score, etc.
            "timestamp": datetime.utcnow().isoformat()
        }
        
        socketio.emit('monte_carlo_progress', update_data, room=room_name)
        logger.info(f"Broadcasted Monte Carlo progress: {completed_sims}/{total_sims} ({progress:.1f}%)")
        
    except Exception as e:
        logger.error(f"Failed to broadcast Monte Carlo progress: {str(e)}")

def broadcast_field_state_update(match_id, field_state, scores=None):
    """Broadcast Push Back field state updates"""
    try:
        room_name = f"field_state_{match_id}"
        
        update_data = {
            "match_id": match_id,
            "field_state": field_state,
            "scores": scores,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        socketio.emit('field_state_update', update_data, room=room_name)
        logger.info(f"Broadcasted field state update for match {match_id}")
        
    except Exception as e:
        logger.error(f"Failed to broadcast field state update: {str(e)}")

def broadcast_match_score_update(match_id, red_score, blue_score, red_breakdown=None, blue_breakdown=None):
    """Broadcast Push Back scoring updates"""
    try:
        room_name = f"match_scoring_{match_id}"
        
        update_data = {
            "match_id": match_id,
            "red_score": red_score,
            "blue_score": blue_score,
            "red_breakdown": red_breakdown,
            "blue_breakdown": blue_breakdown,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        socketio.emit('score_update', update_data, room=room_name)
        logger.info(f"Broadcasted score update for match {match_id}: Red {red_score} - Blue {blue_score}")
        
    except Exception as e:
        logger.error(f"Failed to broadcast score update: {str(e)}")

def broadcast_strategy_recommendation(user_id, robot_specs, recommended_archetype, confidence, reasoning):
    """Broadcast strategy archetype recommendation"""
    try:
        room_name = f"strategy_recommendation_{user_id}"
        
        update_data = {
            "user_id": user_id,
            "robot_specs": robot_specs,
            "recommended_archetype": recommended_archetype,
            "confidence": confidence,
            "reasoning": reasoning,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        socketio.emit('strategy_recommendation', update_data, room=room_name)
        logger.info(f"Broadcasted strategy recommendation for user {user_id}: {recommended_archetype}")
        
    except Exception as e:
        logger.error(f"Failed to broadcast strategy recommendation: {str(e)}")

def broadcast_push_back_system_alert(alert_type, title, message, severity="info"):
    """Broadcast Push Back system alerts"""
    try:
        alert_data = {
            "alert_type": alert_type,  # "analysis_complete", "optimization_ready", "system_update"
            "title": title,
            "message": message,
            "severity": severity,  # "info", "warning", "error", "success"
            "push_back_specific": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        socketio.emit('push_back_alert', alert_data)
        logger.info(f"Broadcasted Push Back alert: {title}")
        
    except Exception as e:
        logger.error(f"Failed to broadcast Push Back alert: {str(e)}")

def broadcast_archetype_analysis_complete(analysis_id, results):
    """Broadcast when archetype analysis is complete"""
    try:
        room_name = f"push_back_analysis_{analysis_id}"
        
        update_data = {
            "analysis_id": analysis_id,
            "event_type": "archetype_analysis_complete",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        socketio.emit('archetype_analysis_complete', update_data, room=room_name)
        logger.info(f"Broadcasted archetype analysis completion for {analysis_id}")
        
    except Exception as e:
        logger.error(f"Failed to broadcast archetype analysis completion: {str(e)}")

@socketio.event
def request_push_back_status():
    """Handle requests for Push Back system status"""
    try:
        # This could call the Push Back system status endpoint
        status_data = {
            "backend_status": "online",
            "analysis_engine_status": "available", 
            "active_analyses": len([room for room in room_subscriptions if "push_back_analysis" in room]),
            "active_simulations": len([room for room in room_subscriptions if "monte_carlo" in room]),
            "push_back_version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        emit('push_back_status', status_data)
        
    except Exception as e:
        logger.error(f"Failed to get Push Back status: {str(e)}")
        emit('error', {"message": "Failed to get Push Back status", "error": str(e)})