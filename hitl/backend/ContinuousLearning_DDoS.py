import os, json, logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =========================================================
# Central Dataset Processor
# =========================================================

class DDoSDataProcessor:
    """Manages central training dataset storage"""

    def __init__(self, central_data_path=None):
        self.central_data_path = central_data_path or os.getenv("CENTRAL_DATA_PATH")
        self._ensure_store()

    def _ensure_store(self):
        if not os.path.exists(self.central_data_path):
            os.makedirs(os.path.dirname(self.central_data_path), exist_ok=True)
            with open(self.central_data_path,'w',encoding='utf-8') as f:
                json.dump([],f,indent=2)
            logger.info(f"✓ Created central dataset → {self.central_data_path}")

    def load_all(self):
        try:
            with open(self.central_data_path,'r',encoding='utf-8') as f:
                data=json.load(f)
            if not isinstance(data, list):
                return []
            return data
        except Exception as e:
            logger.warning(f"Could not load central data: {e}")
            return []

    def append_unique(self,new_samples):
        existing = self.load_all()
        lookup = {self._key(s):s for s in existing}

        added=0
        updated_count=0
        
        for s in new_samples:
            k=self._key(s)
            if k not in lookup:
                lookup[k]=s
                added+=1
            else:
                # Update with newer data if it has corrected labels
                if "actual_action" in s:
                    lookup[k]=s
                    updated_count+=1

        updated=list(lookup.values())
        
        # Keep only last 20k samples to prevent file bloat
        if len(updated) > 20000:
            logger.info(f"Trimming dataset from {len(updated)} to 20000")
            updated = updated[-20000:]
        
        with open(self.central_data_path,'w',encoding='utf-8') as f:
            json.dump(updated,f,indent=2)

        logger.info(f"✓ Added {added} new samples, updated {updated_count} → total {len(updated)}")
        return added

    def _key(self,sample):
        """Generate unique key from history + current features"""
        try:
            # Handle both list of lists and nested structures
            hist = tuple(tuple(x) if isinstance(x, list) else x for x in sample["history"])
            cur = tuple(sample["current"])
            return (hist,cur)
        except Exception as e:
            logger.warning(f"Could not generate key: {e}")
            return str(sample)

    def get_statistics(self):
        """Get dataset statistics"""
        data = self.load_all()
        if not data:
            return {"total": 0, "by_action": {}}
        
        stats = {
            "total": len(data),
            "by_action": {0: 0, 1: 0, 2: 0},  # Allow, RateLimit, Block
            "has_suspicious_score": 0,
            "has_actual_action": 0
        }
        
        for sample in data:
            action = sample.get("action", sample.get("actual_action", 0))
            stats["by_action"][action] = stats["by_action"].get(action, 0) + 1
            
            if "suspicious_score" in sample or "predicted_suspicious" in sample:
                stats["has_suspicious_score"] += 1
            
            if "actual_action" in sample:
                stats["has_actual_action"] += 1
        
        return stats


# =========================================================
# Continuous Learning Manager
# =========================================================

class ContinuousLearningManager:
    """Handles feedback logging and triggers retraining"""

    def __init__(self, feedback_path=None, threshold=20, reload_callback=None):
        self.feedback_path = feedback_path or os.getenv("FEEDBACK_DATA_PATH_DDOS","ddos_feedback.json")
        self.threshold = threshold
        self.reload_callback = reload_callback
        self.processor = DDoSDataProcessor()

    # -----------------------------------------------------

    def log_interaction(self, history, current, ip,
                        predicted_action, predicted_suspicious,
                        actual_action):
        """
        Log human feedback for continuous learning
        
        Args:
            history: List of 6-feature vectors [pps, unique_ips, syn_ratio, rps, bandwidth, always_on]
            current: Single 6-feature vector
            ip: Source IP address
            predicted_action: Model's prediction (0=Allow, 1=RateLimit, 2=Block)
            predicted_suspicious: LSTM suspicious score (0-1)
            actual_action: Human-corrected action (0=Allow, 1=RateLimit, 2=Block)
        """

        # Validate feature dimensions
        if isinstance(current, list):
            if len(current) != 6:
                logger.warning(f"⚠️  Current features should be 6, got {len(current)}. Padding/truncating.")
                if len(current) < 6:
                    current = current + [0] * (6 - len(current))
                    current[5] = 1.0  # always_on flag
                elif len(current) > 6:
                    current = current[:6]

        entry = {
            "history": history,                         # list of 6-feature vectors
            "current": current,                         # single 6-feature vector
            "ip": ip,
            "predicted_action": int(predicted_action),
            "actual_action": int(actual_action),
            "predicted_suspicious": float(predicted_suspicious),
            "suspicious_score": float(predicted_suspicious),  # for LSTM training
            "action": int(actual_action),                     # for XGBoost training
            "timestamp": datetime.utcnow().isoformat(),
            "corrected": predicted_action != actual_action    # track if human corrected
        }

        if not os.path.exists(self.feedback_path):
            data=[entry]
        else:
            try:
                data=json.load(open(self.feedback_path,'r',encoding='utf-8'))
                if not isinstance(data, list):
                    data = [data]
            except:
                data = []
            data.append(entry)

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.feedback_path) if os.path.dirname(self.feedback_path) else '.', exist_ok=True)
        
        json.dump(data,open(self.feedback_path,'w',encoding='utf-8'),indent=2)
        
        action_names = {0: "Allow", 1: "RateLimit", 2: "Block"}
        if entry["corrected"]:
            logger.info(f"✓ Logged CORRECTED feedback → Pred:{action_names[predicted_action]} → Actual:{action_names[actual_action]} (IP: {ip})")
        else:
            logger.info(f"✓ Logged CONFIRMED feedback → {action_names[actual_action]} (IP: {ip})")

    # -----------------------------------------------------

    def get_feedback_count(self):
        if not os.path.exists(self.feedback_path):
            return 0
        try:
            data = json.load(open(self.feedback_path,'r'))
            if isinstance(data, list):
                return len(data)
            return 0
        except:
            return 0

    # -----------------------------------------------------

    def get_feedback_statistics(self):
        """Get statistics about pending feedback"""
        if not os.path.exists(self.feedback_path):
            return {"total": 0, "corrected": 0, "confirmed": 0}
        
        try:
            data = json.load(open(self.feedback_path,'r'))
            if not isinstance(data, list):
                return {"total": 0, "corrected": 0, "confirmed": 0}
            
            corrected = sum(1 for item in data if item.get("corrected", False))
            return {
                "total": len(data),
                "corrected": corrected,
                "confirmed": len(data) - corrected,
                "by_action": {
                    "Allow": sum(1 for item in data if item.get("actual_action") == 0),
                    "RateLimit": sum(1 for item in data if item.get("actual_action") == 1),
                    "Block": sum(1 for item in data if item.get("actual_action") == 2)
                }
            }
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {"total": 0, "corrected": 0, "confirmed": 0}

    # -----------------------------------------------------

    def incorporate_feedback(self):
        """Move feedback to central store and retrain models"""

        if not os.path.exists(self.feedback_path):
            logger.info("ℹ️  No feedback file found")
            return

        try:
            feedback=json.load(open(self.feedback_path,'r',encoding='utf-8'))
            if not isinstance(feedback, list):
                feedback = [feedback] if feedback else []
        except Exception as e:
            logger.error(f"Could not load feedback: {e}")
            return

        if not feedback:
            logger.info("ℹ️  No feedback to incorporate")
            return

        logger.info(f"📊 Processing {len(feedback)} feedback samples")

        # Remove duplicates while keeping most recent
        unique={}
        for item in feedback:
            try:
                key=(tuple(tuple(x) if isinstance(x, list) else x for x in item["history"]),
                     tuple(item["current"]))
                unique[key]=item  # Later items override earlier ones
            except Exception as e:
                logger.warning(f"Could not process item: {e}")

        new_samples=list(unique.values())
        logger.info(f"📊 Unique feedback samples after deduplication: {len(new_samples)}")

        # Show feedback statistics
        stats = self.get_feedback_statistics()
        logger.info(f"📊 Feedback breakdown: {stats['corrected']} corrected, {stats['confirmed']} confirmed")

        # Add to central dataset
        added=self.processor.append_unique(new_samples)

        if added < self.threshold:
            logger.info(f"⚠️  Threshold not met ({added}/{self.threshold}) → skipping retrain")
            # Still clear feedback to avoid reprocessing
            os.remove(self.feedback_path)
            logger.info("✓ Feedback cleared (below threshold)")
            return

        logger.info(f"🚀 Threshold reached ({added} new samples) → starting retraining")

        # Show dataset statistics before retraining
        dataset_stats = self.processor.get_statistics()
        logger.info(f"📊 Central dataset stats: {dataset_stats}")

        # ================= Run Training Pipeline =================

        try:
            import train_lstm
            import train_xgb

            logger.info("="*60)
            logger.info("🔄 STEP 1: Training LSTM (suspicious score detector)")
            logger.info("="*60)
            train_lstm.train()

            logger.info("\n" + "="*60)
            logger.info("🔄 STEP 2: Training XGBoost (action classifier)")
            logger.info("="*60)
            train_xgb.main()

            logger.info("\n" + "="*60)
            logger.info("✅ Training pipeline completed successfully")
            logger.info("="*60)

        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Clear feedback file after successful training
        try:
            os.remove(self.feedback_path)
            logger.info("✓ Feedback cleared after successful retraining")
        except Exception as e:
            logger.warning(f"Could not clear feedback file: {e}")

        # Reload inference server models
        if self.reload_callback:
            logger.info("🔄 Reloading models in inference server...")
            self.reload_callback()
            logger.info("✓ Models reloaded")


# =========================================================
# Standalone Test
# =========================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DDoS Continuous Learning Manager - Test")
    print("="*60)
    
    manager = ContinuousLearningManager(threshold=5)  # Low threshold for testing
    
    # Show current stats
    print("\n📊 Current Dataset Statistics:")
    stats = manager.processor.get_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\n📊 Current Feedback Statistics:")
    feedback_stats = manager.get_feedback_statistics()
    print(json.dumps(feedback_stats, indent=2))
    
    print("\n" + "="*60)