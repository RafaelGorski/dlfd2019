
import numpy as np


class Iris(object):
        
    def shuffle_records(features, labels):
        idx = np.arange(features.shape[0])
        np.random.seed(42)
        np.random.shuffle(idx)
        return features[idx], labels[idx]

    def convert_to_one_hot(vector, num_classes):
        result = np.zeros(shape=(len(vector), num_classes))
        result[np.arange(len(vector)), vector] = 1
        return result.astype(int)

        
class Exquisite_Corpse(object):
    
    def __init__(self):

        with open('data/frankenstein.txt') as f:
            text = f.read()    
            sentences = re.split(r' *[\.\!\?][\'"\)\]]* *', text)

            for i in range(len(sentences)):
                sentences[i] = sentences[i].replace("\n", " ") + "."

        self.sentences = sentences[1000:2500]
     
    def generate_response(model, lexicon_lookup, idx_seq):
        
        end_of_sent_tokens = [".", "!","/",";","?",":"]
        generated_ending = []

        if len(idx_seq) == 0:
            return [3]

        for word in idx_seq:
            p_next_word = model.predict(np.array(word)[None, None])[0,0]

        while not generated_ending or lexicon_lookup[next_word] not in end_of_sent_tokens:
            next_word = np.random.choice(a=p_next_word.shape[-1], p=p_next_word)

            if next_word != 1:
                generated_ending.append(next_word)
                p_next_word = model.predict(np.array(next_word)[None, None])[0,0]

        model.reset_states()
        
        generated_ending = " ".join( [lexicon_lookup[word] if word in lexicon_lookup else "" for word in generated_ending] )
        
        return generated_ending
 




    def text_to_tokens(lines, encoder):
        tokens = [ [word.lower_ for word in encoder(line)] for line in lines]
        return tokens
 
    def make_lexicon(token_seqs, min_freq=4):
        token_counts = {}
        for seq in token_seqs:
            for token in seq:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts[token] = 1
        lexicon = [token for token, count in token_counts.items() if count >= min_freq]
        lexicon = {token:idx + 2 for idx, token in enumerate(lexicon)}
        lexicon[u'<UNK>'] = 1
        lexicon_size = len(lexicon)

        print("lexicon sample ({} total items):".format(len(lexicon)))
        print(list(lexicon.items())[400:405])

        return lexicon

    def get_lexicon_lookup(lexicon):
        lexicon_lookup = { idx: lexicon_item for lexicon_item, idx in lexicon.items()}
        lexicon_lookup[0] = ""
        return lexicon_lookup

    def tokens_to_ids(all_tokens, lexicon):
        ids = [[lexicon[token] if token in lexicon else lexicon['<UNK>'] for token in token_line] for token_line in all_tokens]
        return ids

    def test(self):
        answer = self.test()
        return answer * answer
        
    def test2():
        return 20
    
    
    def generate_text(self, model, length, vocab_size, ix_to_char):

        # starting with random character
        ix = [np.random.randint(vocab_size)]
        y_char = [ix_to_char[ix[-1]]]
        X = np.zeros((1, length, vocab_size))
        
        for i in range(length):
            # appending the last predicted character to sequence
            X[0, i, :][ix[-1]] = 1
            print(ix_to_char[ix[-1]], end="")
            ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
            y_char.append(ix_to_char[ix[-1]])
        return ('').join(y_char)

        
    def load_data(self, data_dir, seq_length):
        data = open(data_dir, 'r').read()
        chars = list(set(data))
        VOCAB_SIZE = len(chars)

        print('Data length: {} characters'.format(len(data)))
        print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

        ix_to_char = {ix:char for ix, char in enumerate(chars)}
        char_to_ix = {char:ix for ix, char in enumerate(chars)}

        print(type(seq_length))
        print(type(VOCAB_SIZE))
        
        X = np.zeros( (int(len(data)/seq_length), seq_length, VOCAB_SIZE))
        y = np.zeros( (int(len(data)/seq_length), seq_length, VOCAB_SIZE))
        for i in range(0, int(len(data)/seq_length) ):
            X_sequence = data[i*seq_length:(i+1)*seq_length]
            X_sequence_ix = [char_to_ix[value] for value in X_sequence]
            input_sequence = np.zeros((seq_length, VOCAB_SIZE))
            
            for j in range(seq_length):
                input_sequence[j][X_sequence_ix[j]] = 1.
                X[i] = input_sequence

                y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
                y_sequence_ix = [char_to_ix[value] for value in y_sequence]
                target_sequence = np.zeros((seq_length, VOCAB_SIZE))
                for j in range(seq_length):
                    target_sequence[j][y_sequence_ix[j]] = 1.
                    y[i] = target_sequence
                    return X, y, VOCAB_SIZE, ix_to_char        


class Style_Transfer(object):
    
    def __init__(self,
                 K,
                 width,
                 height):
        
        self.K = K
        self.width = width
        self.height = height
        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.adjustments_needed = None
        self.loss_value = None
        self.gradient_values = None
            
    def to_VGG_format(self, _image):
        _image = np.expand_dims(_image, axis=0)
        _image[:,:,:,0] -= self.VGG_MEAN[0]
        _image[:,:,:,1] -= self.VGG_MEAN[1]
        _image[:,:,:,2] -= self.VGG_MEAN[2]
        _image = _image[:,:,:,::-1]
        return _image

    def from_VGG_format(self, _x):
        _x = _x.reshape( (self.width, self.height, 3) )
        _x = _x[:,:, ::-1]
        _x[:, :, 0] += self.VGG_MEAN[0]
        _x[:, :, 1] += self.VGG_MEAN[1]
        _x[:, :, 2] += self.VGG_MEAN[2]
        _x = np.clip(_x, 0, 255).astype('uint8')
        return _x

    def content_loss(self, _content, _result):
        return self.K.sum(self.K.square(_result - _content))

    def gram_matrix(self, x):
        features = self.K.batch_flatten(self.K.permute_dimensions(x, (2, 0, 1)))
        gram = self.K.dot(features, self.K.transpose(features))
        return gram

    def style_loss(self, _style, _result):
        S = self.gram_matrix(_style)
        C = self.gram_matrix(_result)
        channels = 3
        size = 256 * 256
        return self.K.sum(self.K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    def noisy_loss(self, x):
        a = self.K.square(x[:, :self.width-1, :self.height-1, :] - x[:, 1:, :self.height-1, :])
        b = self.K.square(x[:, :self.width-1, :self.height-1, :] - x[:, :self.width-1, 1:, :])
        return self.K.sum(self.K.pow(a + b, 1.25))


    def setup_gradients_from_loss_chain(self, loss_chain, result_image):
        gradients = self.K.gradients(loss_chain, result_image)
        loss_and_gradients = [loss_chain] + gradients
        # these adjustments_needed is the derivative of the loss calculations
        self.adjustments_needed = self.K.function([result_image], loss_and_gradients)

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape( (1, self.width, self.height, 3) )
        tweaks = self.adjustments_needed([x])
        self.loss_value = tweaks[0]
        self.gradient_values = tweaks[1].flatten().astype('float64')
        return self.loss_value

    def gradients(self, x):
        assert self.loss_value is not None
        gradient_values = np.copy(self.gradient_values)
        self.loss_value = None
        self.gradient_values = None
        return gradient_values
