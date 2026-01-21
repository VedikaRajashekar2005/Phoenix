import os, json, logging, threading, requests
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from ContinuousLearning_DDoS import ContinuousLearningManager

load_dotenv()

# =========================================================
# Config
# =========================================================

class ReviewConfig:
    INTERMEDIATE_JSON_PATH = os.getenv("INTERMEDIATE_JSON_PATH")
    REVIEW_HISTORY_PATH    = os.getenv("REVIEW_HISTORY_PATH", "review_history.json")
    RETRAIN_THRESHOLD      = 20  # Retrain after 20 reviewed samples
    FIREWALL_API_URL       = "http://localhost:8000"

    CORRECT_REWARD = 1.0
    INCORRECT_PENALTY = -0.5


# =========================================================

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger=logging.getLogger(__name__)


# =========================================================
# Human Review Manager
# =========================================================

class DDoSHumanReviewManager:

    def __init__(self):
        self.config=ReviewConfig()
        self.retraining=False
        self.review_history=self._load_review_history()

        self.learning_manager = ContinuousLearningManager(
            threshold=self.config.RETRAIN_THRESHOLD,
            reload_callback=self._reload_models
        )
        
        logger.info("✓ DDoS Human Review Manager initialized")
        logger.info(f"  Retrain threshold: {self.config.RETRAIN_THRESHOLD}")
        logger.info(f"  Intermediate path: {self.config.INTERMEDIATE_JSON_PATH}")

    # -----------------------------------------------------

    def _reload_models(self):
        """Tell inference server to reload trained models"""
        try:
            logger.info(f"🔄 Requesting model reload from {self.config.FIREWALL_API_URL}")
            r=requests.post(f"{self.config.FIREWALL_API_URL}/reload_models", timeout=10)
            if r.status_code==200:
                logger.info("✓ Inference server reloaded models")
                return True
            else:
                logger.warning(f"⚠️  Reload returned status {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Reload failed: {e}")
            return False

    # -----------------------------------------------------

    def _load_review_history(self):
        """Load historical review statistics"""
        if os.path.exists(self.config.REVIEW_HISTORY_PATH):
            try:
                return json.load(open(self.config.REVIEW_HISTORY_PATH,'r'))
            except:
                pass
        return {
            "reviews": [], 
            "total": 0, 
            "correct": 0, 
            "incorrect": 0,
            "by_action": {
                "Allow": 0,
                "RateLimit": 0,
                "Block": 0
            }
        }

    def _save_review_history(self):
        """Persist review statistics"""
        os.makedirs(os.path.dirname(self.config.REVIEW_HISTORY_PATH) if os.path.dirname(self.config.REVIEW_HISTORY_PATH) else '.', exist_ok=True)
        json.dump(self.review_history,
                  open(self.config.REVIEW_HISTORY_PATH,'w'),indent=2)

    # -----------------------------------------------------

    def load_intermediate_samples(self):
        """Load samples waiting for human review"""
        path=self.config.INTERMEDIATE_JSON_PATH
        if not os.path.exists(path):
            logger.warning(f"Intermediate file not found: {path}")
            return []

        try:
            data=json.load(open(path,'r'))
            
            # Handle different data structures
            if isinstance(data,dict):
                data=data.get("samples",[data])
            elif not isinstance(data, list):
                data = [data]

            # Add readable action names
            action_names = {0: "ALLOW", 1: "RATE_LIMIT", 2: "BLOCK"}
            for s in data:
                predicted_action = s.get("predicted_action", 0)
                s["predicted_action_name"] = action_names.get(predicted_action, "UNKNOWN")
                
                # Ensure we have all required fields
                if "needs_review" not in s:
                    s["needs_review"] = True
                if "confidence" not in s:
                    s["confidence"] = 0.5

            logger.info(f"✓ Loaded {len(data)} samples for review")
            return data
            
        except Exception as e:
            logger.error(f"Error loading intermediate samples: {e}")
            return []

    # -----------------------------------------------------

    def process_review(self, sample, is_correct, corrected_action=None):
        """
        Process human review of a prediction
        
        Args:
            sample: The prediction sample being reviewed
            is_correct: Boolean - was the prediction correct?
            corrected_action: If incorrect, the correct action (0/1/2)
        """

        predicted = sample.get("predicted_action", 0)

        if is_correct:
            final_action = predicted
            reward = self.config.CORRECT_REWARD
            logger.info(f"✓ Review: CONFIRMED {['Allow', 'RateLimit', 'Block'][predicted]}")
        else:
            final_action = corrected_action if corrected_action is not None else 0
            reward = self.config.INCORRECT_PENALTY
            logger.info(f"✓ Review: CORRECTED {['Allow', 'RateLimit', 'Block'][predicted]} → {['Allow', 'RateLimit', 'Block'][final_action]}")

        # Validate features
        current = sample.get("current", [])
        if len(current) != 6:
            logger.warning(f"⚠️  Sample has {len(current)} features, expected 6. Adjusting...")
            if len(current) < 6:
                current = current + [0] * (6 - len(current))
                current[5] = 1.0  # always_on flag
            elif len(current) > 6:
                current = current[:6]

        # Create reviewed entry
        reviewed = {
            "history": sample.get("history", []),
            "current": current,
            "ip": sample.get("ip", "unknown"),
            "predicted_action": predicted,
            "predicted_suspicious": sample.get("predicted_suspicious", 0.0),
            "actual_action": final_action,
            "reward": reward,
            "timestamp": datetime.utcnow().isoformat(),
            "corrected": not is_correct
        }

        # Update statistics
        self.review_history["reviews"].append(reviewed)
        self.review_history["total"] += 1
        
        action_names = {0: "Allow", 1: "RateLimit", 2: "Block"}
        action_name = action_names[final_action]
        
        if is_correct:
            self.review_history["correct"] += 1
        else:
            self.review_history["incorrect"] += 1
        
        self.review_history["by_action"][action_name] = self.review_history["by_action"].get(action_name, 0) + 1

        # Keep only last 1000 reviews in history
        if len(self.review_history["reviews"]) > 1000:
            self.review_history["reviews"] = self.review_history["reviews"][-1000:]

        self._save_review_history()

        # Log feedback for continuous learning
        self.learning_manager.log_interaction(
            history=reviewed["history"],
            current=reviewed["current"],
            ip=reviewed["ip"],
            predicted_action=predicted,
            predicted_suspicious=reviewed["predicted_suspicious"],
            actual_action=final_action
        )

        # Check if we should trigger retraining
        self._auto_retrain_check()

        return {
            "success": True,
            "final_action": final_action,
            "final_action_name": action_name,
            "reward": reward,
            "pending_retrains": self.learning_manager.get_feedback_count()
        }

    # -----------------------------------------------------

    def _auto_retrain_check(self):
        """Check if we should trigger automatic retraining"""
        if self.retraining:
            logger.info("⚠️  Retraining already in progress, skipping check")
            return
        
        count = self.learning_manager.get_feedback_count()
        logger.info(f"📊 Feedback count: {count}/{self.config.RETRAIN_THRESHOLD}")
        
        if count >= self.config.RETRAIN_THRESHOLD:
            logger.info(f"🚀 Threshold reached! Starting automatic retraining...")
            t = threading.Thread(target=self._execute_retraining)
            t.daemon = True
            t.start()

    def _execute_retraining(self):
        """Execute the retraining process in background"""
        self.retraining = True
        logger.info("="*60)
        logger.info("🔄 AUTO RETRAINING STARTED")
        logger.info("="*60)
        
        try:
            self.learning_manager.incorporate_feedback()
            logger.info("="*60)
            logger.info("✅ AUTO RETRAINING COMPLETE")
            logger.info("="*60)
        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.retraining = False

    # -----------------------------------------------------

    def get_statistics(self):
        """Get comprehensive statistics"""
        feedback_stats = self.learning_manager.get_feedback_statistics()
        dataset_stats = self.learning_manager.processor.get_statistics()
        
        return {
            "review_history": {
                "total_reviews": self.review_history["total"],
                "correct": self.review_history["correct"],
                "incorrect": self.review_history["incorrect"],
                "accuracy": (self.review_history["correct"] / max(1, self.review_history["total"])) * 100,
                "by_action": self.review_history.get("by_action", {})
            },
            "pending_feedback": feedback_stats,
            "central_dataset": dataset_stats,
            "retraining": {
                "in_progress": self.retraining,
                "threshold": self.config.RETRAIN_THRESHOLD,
                "progress": f"{feedback_stats['total']}/{self.config.RETRAIN_THRESHOLD}"
            }
        }


# =========================================================
# Flask API
# =========================================================

app = Flask(__name__)
CORS(app)

manager = DDoSHumanReviewManager()

@app.route("/load_samples", methods=["GET"])
def load_samples():
    """Get all samples needing review"""
    samples = manager.load_intermediate_samples()
    return jsonify({
        "success": True,
        "samples": samples,
        "count": len(samples)
    })

@app.route("/review", methods=["POST"])
def review():
    """Submit a human review"""
    try:
        data = request.get_json()
        
        if not data or "sample" not in data:
            return jsonify({"success": False, "error": "Missing sample data"}), 400
        
        is_correct = data.get("is_correct", False)
        corrected_action = data.get("corrected_action")
        
        result = manager.process_review(
            sample=data["sample"],
            is_correct=is_correct,
            corrected_action=corrected_action
        )
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing review: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/statistics", methods=["GET"])
def statistics():
    """Get system statistics"""
    try:
        stats = manager.get_statistics()
        return jsonify({"success": True, "statistics": stats})
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    feedback_count = manager.learning_manager.get_feedback_count()
    return jsonify({
        "status": "running",
        "retraining": manager.retraining,
        "feedback_count": feedback_count,
        "threshold": manager.config.RETRAIN_THRESHOLD
    })

@app.route("/trigger_retrain", methods=["POST"])
def trigger_retrain():
    """Manually trigger retraining"""
    if manager.retraining:
        return jsonify({"success": False, "message": "Retraining already in progress"})
    
    feedback_count = manager.learning_manager.get_feedback_count()
    if feedback_count == 0:
        return jsonify({"success": False, "message": "No feedback available for retraining"})
    
    logger.info("🔄 Manual retrain triggered")
    t = threading.Thread(target=manager._execute_retraining)
    t.daemon = True
    t.start()
    
    return jsonify({
        "success": True,
        "message": f"Retraining started with {feedback_count} samples"
    })

# =========================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DDoS Human Review API Server")
    print("="*60)
    print(f"Retrain Threshold: {manager.config.RETRAIN_THRESHOLD} samples")
    print(f"Intermediate Path: {manager.config.INTERMEDIATE_JSON_PATH}")
    print(f"Review History: {manager.config.REVIEW_HISTORY_PATH}")
    print("="*60)
    print("Running on http://localhost:5002")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5002, debug=False)