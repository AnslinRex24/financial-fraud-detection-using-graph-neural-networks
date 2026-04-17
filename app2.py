# Fraud Detection Dashboard - Flask Application
# Complete system with inference, visualization, and explanation

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import HeteroGraphConv, HeteroEmbedding, GraphConv
from torch.nn import Linear
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import networkx as nx
import os
import traceback

# =============================================================================
# MODEL CLASSES (Must match training code)
# =============================================================================

class FFBlock(nn.Module):
    """Feed-forward block"""
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
        super().__init__()
        self.input_layer = Linear(in_dim, hidden_dim)
        self.hidden_layer = Linear(hidden_dim, hidden_dim)
        self.output_layer = Linear(hidden_dim, out_dim)
        self.n_layers = n_layers
    
    def forward(self, in_feats):
        h = self.input_layer(in_feats)
        h = F.relu(h)
        for i in range(1, self.n_layers):
            h = self.hidden_layer(h)
            h = F.relu(h)
        h = self.output_layer(h)
        return h

class EnhancedRGCN(nn.Module):
    """Enhanced RGCN model"""
    def __init__(self, target_feature_dim, config, graph, node_types):
        super().__init__()
        self.config = config
        self.node_types = [nt for nt in node_types if nt in graph.ntypes]
        
        entry_module_dict = {
            etype: GraphConv(config['input_dim'], config['hidden_dim']) 
            for etype in graph.etypes
        }
        hidden_module_dict = {
            etype: GraphConv(config['hidden_dim'], config['hidden_dim']) 
            for etype in graph.etypes
        }
        final_model_dict1 = {
            etype: GraphConv(config['hidden_dim'], config['target_out_dim']) 
            for src, etype, dst in graph.canonical_etypes if dst == 'target'
        }
        final_model_dict2 = {
            etype: GraphConv(config['hidden_dim'], 1) 
            for src, etype, dst in graph.canonical_etypes if dst != 'target'
        }
        final_model_dict = {**final_model_dict1, **final_model_dict2}
        
        num_embeddings_dict = {
            src: graph.num_nodes(src) 
            for src, etype, dst in graph.canonical_etypes 
            if dst == 'target' and src != 'target'
        }
        
        if num_embeddings_dict:
            self.embed_layer = HeteroEmbedding(num_embeddings_dict, config['input_dim'])
        else:
            self.embed_layer = None
        
        self.target_preprocessing = FFBlock(
            target_feature_dim, 
            config['target_preprocessing_hidden_dim'], 
            config['input_dim'],
            config['target_preprocessing_layers']
        )
        self.conv1 = HeteroGraphConv(entry_module_dict, aggregate='sum')
        self.conv2 = HeteroGraphConv(hidden_module_dict, aggregate='sum')
        self.conv3 = HeteroGraphConv(final_model_dict, aggregate='sum')
        self.target_postprocessing = FFBlock(
            config['target_out_dim'],
            config['target_postprocessing_hidden_dim'],
            1,
            config['target_postprocessing_layers']
        )
        self.num_conv_layers = config['conv_layers']
        
    def forward(self, graph, input_features, return_embeddings=False):
        if self.embed_layer is not None and self.node_types:
            embeds = self.embed_layer({
                ntype: graph.nodes(ntype) for ntype in self.node_types
            })
        else:
            embeds = {}
        
        target_features = self.target_preprocessing(input_features.float())
        embeds['target'] = target_features
        
        h1 = self.conv1(graph, embeds)
        h1 = {k: F.relu(v) for k, v in h1.items()}
        
        h2 = h1
        for i in range(2, self.num_conv_layers):
            h2 = self.conv2(graph, h2)
            h2 = {k: F.relu(v) for k, v in h2.items()}
        
        h3 = self.conv3(graph, h2)
        
        if 'target' in h3:
            h3['target'] = self.target_postprocessing(h3['target'])
        
        if return_embeddings:
            return h3, {'layer1': h1, 'layer2': h2, 'layer3': h3}
        
        return h3

# =============================================================================
# INFERENCE ENGINE
# =============================================================================

class FraudDetectionInference:
    """Handle model inference and predictions"""
    
    def __init__(self, system_path='fraud_detection_complete_system.pkl'):
        print("Loading fraud detection system...")
        
        try:
            # Load saved system
            with open(system_path, 'rb') as f:
                self.saved_system = pickle.load(f)
            
            self.config = self.saved_system['config']
            self.graph_info = self.saved_system['graph_info']
            self.data_info = self.saved_system['data_info']
            self.feature_names = self.data_info['feature_names']
            self.id_to_node = self.data_info['id_to_node']
            
            # Load baseline models
            self.rf_model = self.saved_system['baseline_models']['random_forest']
            self.scaler = self.saved_system['baseline_models']['scaler']
            
            print("System loaded successfully!")
        except Exception as e:
            print(f"Error loading system: {str(e)}")
            raise
        
    def load_gnn_model(self, transactions_df, identity_df):
        """Rebuild graph and load GNN model"""
        
        try:
            # Store transactions_df for later use
            self.transactions_df = transactions_df
            
            # Prepare data
            id_cols = ['card1','card2','card3','card4','card5','card6','ProductCD',
                       'addr1','addr2','P_emaildomain','R_emaildomain']
            cat_cols = ['M1','M2','M3','M4','M5','M6','M7','M8','M9']
            
            transactions_non_features = ['isFraud','TransactionDT'] + id_cols
            features_cols = [col for col in transactions_df.columns if col not in transactions_non_features]
            
            features_df = pd.get_dummies(transactions_df[features_cols], columns=cat_cols).fillna(0)
            features_df['TransactionAmt'] = features_df['TransactionAmt'].apply(np.log10)
            
            node_types = id_cols + list(identity_df.columns)
            node_types.remove('TransactionID')
            
            full_identity_df = identity_df.merge(
                transactions_df[id_cols + ['TransactionID']], 
                on='TransactionID', how='right'
            )
            
            edge_dfs = {}
            for ntype in node_types:
                edge_dfs[ntype] = full_identity_df[['TransactionID', ntype]].dropna()
            
            # Build graph
            edgelists = {}
            num_nodes_dict = {}
            
            for ntype in node_types:
                if ntype not in edge_dfs or len(edge_dfs[ntype]) == 0:
                    continue
                    
                edge_type = ('target', f'target<>{ntype}', ntype)
                rev_edge_type = (ntype, f'{ntype}<>target', 'target')
                
                source_nodes = edge_dfs[ntype]['TransactionID'].apply(
                    lambda a: self.id_to_node['target'][a]
                ).to_numpy()
                
                destination_nodes = edge_dfs[ntype][ntype].apply(
                    lambda a: self.id_to_node[ntype][a]
                ).to_numpy()
                
                edgelists[edge_type] = (source_nodes, destination_nodes)
                edgelists[rev_edge_type] = (destination_nodes, source_nodes)
                num_nodes_dict[ntype] = len(np.unique(destination_nodes))
            
            all_target_nodes = np.arange(len(transactions_df))
            edgelists[('target','target<>target','target')] = (all_target_nodes, all_target_nodes)
            num_nodes_dict['target'] = len(transactions_df)
            
            graph = dgl.heterograph(edgelists, num_nodes_dict)
            
            # Add features
            feature_tensor = torch.from_numpy(features_df.drop('TransactionID', axis=1).to_numpy()).float()
            graph.nodes['target'].data['features'] = feature_tensor
            
            # Normalize
            mean = torch.mean(graph.nodes['target'].data['features'], axis=0)
            std = torch.sqrt(torch.sum((graph.nodes['target'].data['features'] - mean)**2, axis=0) / 
                             graph.nodes['target'].data['features'].shape[0])
            graph.nodes['target'].data['features'] = (graph.nodes['target'].data['features'] - mean) / std
            
            # Load model
            target_feature_dim = self.saved_system['target_feature_dim']
            model = EnhancedRGCN(target_feature_dim, self.config, graph, node_types)
            model.load_state_dict(self.saved_system['rgcn_model_state'])
            model.eval()
            
            self.graph = graph
            self.model = model
            
            print(f"GNN model loaded successfully with {len(transactions_df)} transactions")
            
            return graph, model
        except Exception as e:
            print(f"Error in load_gnn_model: {str(e)}")
            traceback.print_exc()
            raise
    
    def predict_single(self, transaction_id):
        """Predict fraud for a single transaction"""
        
        try:
            if transaction_id not in self.id_to_node['target']:
                return {'status': 'error', 'message': f'Transaction ID {transaction_id} not found in dataset'}
            
            node_idx = self.id_to_node['target'][transaction_id]
            
            with torch.no_grad():
                features = self.graph.nodes['target'].data['features']
                logits = self.model(self.graph, features)['target']
                prob = torch.sigmoid(logits[node_idx]).item()
            
            # Feature importance
            feature_importance = self.get_feature_importance(node_idx)
            
            # Connected nodes
            connected_nodes = self.get_connected_nodes(node_idx)
            
            prediction = 'Fraud' if prob > 0.5 else 'Legitimate'
            confidence = max(prob, 1-prob)
            
            explanation = f"Transaction {transaction_id} predicted as {prediction} "
            explanation += f"with {confidence:.1%} confidence.\n\n"
            
            if feature_importance:
                explanation += "Top contributing features:\n"
                for feature, info in list(feature_importance.items())[:5]:
                    explanation += f"• {feature}: {info['value']:.3f} (importance: {info['importance']:.3f})\n"
            
            return {
                'status': 'success',
                'transaction_id': transaction_id,
                'fraud_probability': prob,
                'prediction': prediction,
                'confidence': confidence,
                'feature_importance': feature_importance,
                'connected_nodes': connected_nodes,
                'explanation_text': explanation
            }
        except Exception as e:
            print(f"Error in predict_single: {str(e)}")
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
    
    def predict_from_features(self, features_dict):
        """Predict fraud from manual feature input"""
        
        try:
            # Convert features to proper format
            feature_vector = np.zeros(len(self.feature_names))
            
            for i, feature_name in enumerate(self.feature_names):
                if feature_name in features_dict:
                    features_dict[feature_name]
                    feature_vector[i] = features_dict[feature_name]
            
            # Use Random Forest for manual predictions (simpler)
            feature_vector_scaled = self.scaler.transform([feature_vector])
            prob = self.rf_model.predict_proba(feature_vector_scaled)[0][1]
            
            prediction = 'Fraud' if prob > 0.5 else 'Legitimate'
            confidence = max(prob, 1-prob)
            
            # Get feature importance from RF
            feature_importance = {}
            importances = self.rf_model.feature_importances_
            top_indices = np.argsort(importances)[-10:][::-1]
            
            for idx in top_indices:
                if idx < len(self.feature_names):
                    feature_importance[self.feature_names[idx]] = {
                        'importance': float(importances[idx]),
                        'value': float(feature_vector[idx])
                    }
            
            explanation = f"Prediction: {prediction} with {confidence:.1%} confidence\n\n"
            explanation += "Top contributing features:\n"
            for feature, info in list(feature_importance.items())[:5]:
                explanation += f"• {feature}: {info['value']:.3f} (importance: {info['importance']:.3f})\n"
            
            return {
                'status': 'success',
                'fraud_probability': prob,
                'prediction': prediction,
                'confidence': confidence,
                'feature_importance': feature_importance,
                'explanation_text': explanation
            }
        except Exception as e:
            print(f"Error in predict_from_features: {str(e)}")
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
    
    def get_feature_importance(self, node_idx):
        """Get feature importance for a transaction"""
        
        try:
            node_features = self.graph.nodes['target'].data['features'][node_idx:node_idx+1]
            node_features.requires_grad_(True)
            
            logits = self.model.target_postprocessing(
                self.model.target_preprocessing(node_features)
            )
            prob = torch.sigmoid(logits)
            prob.backward()
            
            importance = torch.abs(node_features.grad).squeeze().detach().numpy()
            top_indices = np.argsort(importance)[-10:][::-1]
            
            feature_importance = {}
            for i, idx in enumerate(top_indices):
                if idx < len(self.feature_names):
                    feature_importance[self.feature_names[idx]] = {
                        'importance': float(importance[idx]),
                        'value': float(node_features[0, idx].detach().numpy()),
                        'rank': i + 1
                    }
            
            return feature_importance
        except Exception as e:
            print(f"Error in get_feature_importance: {str(e)}")
            return {}
    
    def get_connected_nodes(self, node_idx):
        """Get connected nodes information"""
        
        try:
            connected_info = {}
            
            for ntype in self.model.node_types[:10]:
                edge_type = f'target<>{ntype}'
                if edge_type in self.graph.etypes:
                    edges = self.graph.edges(etype=edge_type)
                    source_mask = edges[0] == node_idx
                    
                    if source_mask.any():
                        connected_nodes = edges[1][source_mask]
                        connected_info[ntype] = len(connected_nodes.unique())
            
            return connected_info
        except Exception as e:
            print(f"Error in get_connected_nodes: {str(e)}")
            return {}
    
    def create_transaction_graph(self, transaction_id, max_neighbors=20):
        """Create graph visualization for a transaction"""
        
        try:
            if transaction_id not in self.id_to_node['target']:
                return None
            
            node_idx = self.id_to_node['target'][transaction_id]
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Get prediction for color
            with torch.no_grad():
                features = self.graph.nodes['target'].data['features']
                logits = self.model(self.graph, features)['target']
                prob = torch.sigmoid(logits[node_idx]).item()
            
            # Add center node
            is_fraud = prob > 0.5
            G.add_node(f"T_{transaction_id}", 
                      node_type='transaction', 
                      is_fraud=is_fraud,
                      probability=prob)
            
            # Add connected nodes
            node_count = 1
            for ntype in self.model.node_types:
                if node_count >= max_neighbors:
                    break
                    
                edge_type = f'target<>{ntype}'
                if edge_type in self.graph.etypes:
                    edges = self.graph.edges(etype=edge_type)
                    source_mask = edges[0] == node_idx
                    
                    if source_mask.any():
                        connected_nodes = edges[1][source_mask][:5]
                        
                        for conn_node in connected_nodes:
                            if node_count >= max_neighbors:
                                break
                            node_name = f"{ntype}_{conn_node}"
                            G.add_node(node_name, node_type=ntype)
                            G.add_edge(f"T_{transaction_id}", node_name)
                            node_count += 1
            
            # Create plotly visualization
            pos = nx.spring_layout(G, k=0.5, iterations=50)
            
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines')
            
            node_x = []
            node_y = []
            node_colors = []
            node_text = []
            node_sizes = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                if G.nodes[node]['node_type'] == 'transaction':
                    if G.nodes[node]['is_fraud']:
                        node_colors.append('red')
                        node_text.append(f"Transaction {transaction_id}<br>FRAUD ({G.nodes[node]['probability']:.2%})")
                    else:
                        node_colors.append('green')
                        node_text.append(f"Transaction {transaction_id}<br>LEGITIMATE ({G.nodes[node]['probability']:.2%})")
                    node_sizes.append(30)
                else:
                    node_colors.append('lightblue')
                    node_text.append(f"{node}")
                    node_sizes.append(15)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=[n.split('_')[0][:10] for n in G.nodes()],
                hovertext=node_text,
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=2, color='white')))
            
            fig = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                            title=f'Transaction Network - {transaction_id}',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )
            
            return json.dumps(fig, cls=PlotlyJSONEncoder)
        except Exception as e:
            print(f"Error in create_transaction_graph: {str(e)}")
            traceback.print_exc()
            return None

# =============================================================================
# FLASK APPLICATION
# =============================================================================

app = Flask(__name__)

# Initialize inference engine
inference_engine = None

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('indexx.html')

@app.route('/api/init', methods=['POST'])
def initialize_system():
    """Initialize the fraud detection system"""
    global inference_engine
    
    try:
        # Check if required files exist
        required_files = [
            'fraud_detection_complete_system.pkl',
            'train_transaction.csv',
            'train_identity.csv'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            return jsonify({
                'status': 'error',
                'message': f'Missing required files: {", ".join(missing_files)}'
            }), 500
        
        if inference_engine is None:
            inference_engine = FraudDetectionInference()
            
            # Load original dataset for GNN
            print("Loading transaction and identity data...")
            transactions_df = pd.read_csv('train_transaction.csv')
            identity_df = pd.read_csv('train_identity.csv')
            
            print(f"Loaded {len(transactions_df)} transactions")
            
            inference_engine.load_gnn_model(transactions_df, identity_df)
        
        return jsonify({
            'status': 'success',
            'message': 'System initialized successfully',
            'num_transactions': len(inference_engine.transactions_df),
            'model_performance': inference_engine.saved_system['results']['best_model_performance']
        })
    except MemoryError as e:
        return jsonify({
            'status': 'error',
            'message': 'Insufficient memory to load the model. Please try with a smaller dataset or increase available RAM.'
        }), 500
    except FileNotFoundError as e:
        return jsonify({
            'status': 'error',
            'message': f'File not found: {str(e)}'
        }), 500
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error during initialization:\n{error_details}")
        return jsonify({
            'status': 'error',
            'message': f'Initialization failed: {str(e)}'
        }), 500

@app.route('/api/predict_transaction', methods=['POST'])
def predict_transaction():
    """Predict fraud for a specific transaction ID"""
    global inference_engine
    
    if inference_engine is None:
        return jsonify({'status': 'error', 'message': 'System not initialized. Please click "Initialize System" first.'}), 400
    
    data = request.json
    transaction_id = data.get('transaction_id')
    
    if not transaction_id:
        return jsonify({'status': 'error', 'message': 'Transaction ID required'}), 400
    
    try:
        transaction_id = int(transaction_id)
        result = inference_engine.predict_single(transaction_id)
        
        if result.get('status') == 'error':
            return jsonify(result), 400
        
        # Get graph visualization
        graph_json = inference_engine.create_transaction_graph(transaction_id)
        if graph_json:
            result['graph'] = graph_json
        
        return jsonify(result)
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Transaction ID must be a valid number'}), 400
    except Exception as e:
        print(f"Error in predict_transaction: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict_manual', methods=['POST'])
def predict_manual():
    """Predict fraud from manual feature input"""
    global inference_engine
    
    if inference_engine is None:
        return jsonify({'status': 'error', 'message': 'System not initialized. Please click "Initialize System" first.'}), 400
    
    data = request.json
    features = data.get('features')
    
    if not features:
        return jsonify({'status': 'error', 'message': 'Features required'}), 400
    
    try:
        result = inference_engine.predict_from_features(features)
        return jsonify(result)
    except Exception as e:
        print(f"Error in predict_manual: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Predict fraud for uploaded CSV file"""
    global inference_engine
    
    if inference_engine is None:
        return jsonify({'status': 'error', 'message': 'System not initialized. Please click "Initialize System" first.'}), 400
    
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'status': 'error', 'message': 'File must be a CSV'}), 400
    
    try:
        # Read CSV
        df = pd.read_csv(file)
        
        # Process each transaction
        results = []
        for idx, row in df.iterrows():
            if idx >= 10:  # Limit to first 10 transactions
                break
            
            # Convert row to features dict
            features = {}
            for col in row.index:
                if col in inference_engine.feature_names:
                    features[col] = float(row[col]) if not pd.isna(row[col]) else 0.0
            
            # Predict
            result = inference_engine.predict_from_features(features)
            if result.get('status') == 'success':
                result['row_index'] = idx
                results.append(result)
        
        return jsonify({
            'status': 'success',
            'total_rows': len(df),
            'processed_rows': len(results),
            'results': results
        })
    
    except Exception as e:
        print(f"Error in upload_csv: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/get_feature_list', methods=['GET'])
def get_feature_list():
    """Get list of available features"""
    global inference_engine
    
    if inference_engine is None:
        return jsonify({'status': 'error', 'message': 'System not initialized'}), 400
    
    # Return important features grouped by category
    features = {
        'transaction': ['TransactionAmt', 'ProductCD'],
        'card': ['card1', 'card2', 'card3', 'card4', 'card5', 'card6'],
        'address': ['addr1', 'addr2'],
        'distance': ['dist1', 'dist2'],
        'counts': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14'],
        'time_deltas': ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10'],
        'email': ['P_emaildomain', 'R_emaildomain']
    }
    
    return jsonify({'features': features})

if __name__ == '__main__':
    print("Starting Fraud Detection Dashboard...")
    print("Make sure the following files are in the same directory:")
    print("- fraud_detection_complete_system.pkl")
    print("- train_transaction.csv")
    print("- train_identity.csv")
    print("\nDashboard will be available at: http://localhost:8001")
    
    app.run(debug=True, host='0.0.0.0', port=8001)